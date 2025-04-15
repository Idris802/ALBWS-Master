#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import open3d as o3d
import rospkg
import os
import tf
import tf.transformations as tfm
import time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO
from scipy import ndimage


class WeedDetectorRoot:
    def __init__(self):
        rospy.init_node("weed_detector", anonymous=True)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.camera_matrix_depth = None

        rospack = rospkg.RosPack()
        package_path = rospack.get_path("moveit_ur3")
        model_path = os.path.join(package_path, "models", "best_20.pt")
        self.model = YOLO(model_path)

        self.color_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)
        self.caemera_info_depth_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, 
                                                        self.camera_info_depth_callback)
        self.latest_depth = None
        self.listener = tf.TransformListener()

        self.weed_pub = rospy.Publisher("/weed_coordinates", PointStamped, queue_size=10)

        self.preprocess_times = []
        self.inference_times = []
        self.postprocess_times = []
        self.num_weeds_list = []
        self.average_interval = 500

    def camera_info_depth_callback(self, msg):
        """ Store camera intrinsics from /camera/color/camera_info. """
        fx = msg.K[0]
        fy = msg.K[4]
        cx = msg.K[2]
        cy = msg.K[5]
        self.camera_matrix_depth = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])
        rospy.loginfo_once("Camera intrinsics: fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}".format(fx, fy, cx, cy))

    def camera_info_callback(self, msg):
        """ Store camera intrinsics from /camera/color/camera_info. """
        fx = msg.K[0]
        fy = msg.K[4]
        cx = msg.K[2]
        cy = msg.K[5]
        self.camera_matrix = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])
        rospy.loginfo_once("Camera intrinsics: fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}".format(fx, fy, cx, cy))


    def pixel_to_world(self, u, v, depth):
        if self.camera_matrix is None:
            rospy.logwarn("Camera intrinsics not received yet!")
            return None, None, None

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        return X, Y, Z

    def transform_camera_to_base(self, x_cam, y_cam, z_cam):
        try:
            self.listener.waitForTransform('/base_link', '/camera_depth_optical_frame', rospy.Time(0), rospy.Duration(3.0))
            (trans, rot) = self.listener.lookupTransform('/base_link', '/camera_depth_optical_frame', rospy.Time(0))

            cam_point = np.array([x_cam, y_cam, z_cam, 1])
            transform_matrix = tfm.concatenate_matrices(tfm.translation_matrix(trans), tfm.quaternion_matrix(rot))
            base_point = np.dot(transform_matrix, cam_point)
            return base_point[0], base_point[1], base_point[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("TF transform lookup failed!")
            return None, None, None

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def ransac(self, d, x1, y1):
        if d.dtype != np.float32:
            d = d.astype(np.float32)
        d_meters = d / 1000.0
        depth_o3d = o3d.geometry.Image(d_meters)
        width = d_meters.shape[1] 
        height = d_meters.shape[0] 
        fx = self.camera_matrix_depth[0, 0]
        fy = self.camera_matrix_depth[1, 1]
        cx = self.camera_matrix_depth[0, 2] - x1 
        cy = self.camera_matrix_depth[1, 2] - y1
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        depth_scale = 1
        depth_trunc = 1.0
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsic,
                                                              depth_scale=depth_scale,
                                                              depth_trunc=depth_trunc, 
                                                              stride=1)
        distance_threshold = 0.004 
        ransac_n = 3             
        num_iterations = 1000

        plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
        [a, b, c, d_coeff] = plane_model

        ground_plane = pcd.select_by_index(inliers)
        ground_plane.paint_uniform_color([0.0, 1.0, 0.0]) 
        weed_cloud = pcd.select_by_index(inliers, invert=True)
        weed_cloud.paint_uniform_color([1.0, 0.0, 0.0])  
        weed_points = np.asarray(weed_cloud.points)
        
        #o3d.visualization.draw_geometries([ground_plane, weed_cloud])

        distances = np.abs(a * weed_points[:, 0] + b * weed_points[:, 1] +
                    c * weed_points[:, 2] + d_coeff)

        stem_index = np.argmin(distances)
        stem_point = weed_points[stem_index] 
        x_base, y_base, z_base = self.transform_camera_to_base(stem_point[0], 
                                                               stem_point[1],
                                                               stem_point[2])

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(stem_point)
        sphere.paint_uniform_color([0.0, 0.0, 1.0])  

        #o3d.visualization.draw_geometries([weed_cloud, sphere])

        return x_base, y_base, z_base

    def color_pixel_to_depth_pixel(self, u_color, v_color, depth_value, K_color, K_depth):
        self.listener.waitForTransform('/camera_depth_optical_frame', '/camera_color_optical_frame', rospy.Time(0), rospy.Duration(3.0))
        (trans, rot) = self.listener.lookupTransform('/camera_depth_optical_frame', '/camera_color_optical_frame', rospy.Time(0))
        R = tfm.quaternion_matrix(rot)[:3, :3] 
        T = np.array(trans)  
        x_norm = (u_color - K_color[0,2]) / K_color[0,0]
        y_norm = (v_color - K_color[1,2]) / K_color[1,1]

        point_color = np.array([x_norm * depth_value, y_norm * depth_value, depth_value])
        point_depth = R.dot(point_color) + T
        
        u_depth = (K_depth[0,0] * point_depth[0] / point_depth[2]) + K_depth[0,2]
        v_depth = (K_depth[1,1] * point_depth[1] / point_depth[2]) + K_depth[1,2]
        
        return int(round(u_depth)), int(round(v_depth))

    def image_callback(self, msg):
            if self.latest_depth is None:
                rospy.logwarn("No depth data yet, skipping detection.")
                return

            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            results = self.model(cv_image)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(cv_image.shape[1], x2); y2 = min(cv_image.shape[0], y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    u = (x1 + x2) // 2 
                    v = (y1 + y2) // 2 

                    depth_value = self.latest_depth[v, u]
                    if depth_value > 10:
                        depth_value /= 1000.0 
                    if depth_value <= 0:
                        rospy.logwarn("Invalid depth at weed centroid.")
                        continue

                    corners_color = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                    corners_depth = [self.color_pixel_to_depth_pixel(u, v, depth_value,
                                                self.camera_matrix, self.camera_matrix_depth,)
                                    for (u, v) in corners_color]
                    
                    us, vs = zip(*corners_depth)
                    u_depth_min, u_depth_max = min(us), max(us)
                    v_depth_min, v_depth_max = min(vs), max(vs)
                    roi_depth_aligned = self.latest_depth[v_depth_min:v_depth_max, u_depth_min:u_depth_max].copy()
                    x_base, y_base, z_base = self.ransac(roi_depth_aligned, u_depth_min, v_depth_min)

                    #X_cam, Y_cam, Z_cam = self.pixel_to_world(u, v, depth_value)
                    #x_base, y_base, z_base = self.transform_camera_to_base(X_cam, Y_cam, Z_cam)
                    if x_base is not None:
                        rospy.loginfo("Weed root in Base Frame: X={:.3f}, Y={:.3f}, Z={:.3f}".format(x_base, y_base, z_base))
                        weed_msg = PointStamped()
                        weed_msg.header.stamp = rospy.Time.now()
                        weed_msg.header.frame_id = "base_link"
                        weed_msg.point.x = x_base
                        weed_msg.point.y = y_base
                        weed_msg.point.z = z_base
                        self.weed_pub.publish(weed_msg)

                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(cv_image, (u, v), 4, (0, 0, 255), -1)
                    text = "X:{:.2f} Y:{:.2f} Z:{:.2f}".format(x_base, y_base, z_base)
                    cv2.putText(cv_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

                cv2.imshow("Weed Detection", cv_image)
                cv2.waitKey(1)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    node = WeedDetectorRoot()
    node.run()
  