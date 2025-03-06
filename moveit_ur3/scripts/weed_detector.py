#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import rospkg
import os

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
from ultralytics import YOLO

import tf
import tf.transformations as tfm

class WeedDetectorRoot:
    def __init__(self):
        rospy.init_node("weed_detector", anonymous=True)

        self.bridge = CvBridge()
        self.camera_matrix = None

        # Load YOLO Model
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("moveit_ur3")
        model_path = os.path.join(package_path, "models", "best_20.pt")
        self.model = YOLO(model_path)

        # Subscribers
        self.color_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)

        self.latest_depth = None
        self.listener = tf.TransformListener()

        self.weed_pub = rospy.Publisher("/weed_coordinates", PointStamped, queue_size=10)

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
        """ Convert (u, v, depth) to (X, Y, Z) in camera frame using the pinhole model. """
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
        """ Transform a 3D point from camera_color_optical_frame to base_link. """
        try:
            self.listener.waitForTransform('/base_link', '/camera_color_optical_frame', rospy.Time(0), rospy.Duration(3.0))
            (trans, rot) = self.listener.lookupTransform('/base_link', '/camera_color_optical_frame', rospy.Time(0))

            cam_point = np.array([x_cam, y_cam, z_cam, 1])
            transform_matrix = tfm.concatenate_matrices(tfm.translation_matrix(trans), tfm.quaternion_matrix(rot))
            base_point = np.dot(transform_matrix, cam_point)
            return base_point[0], base_point[1], base_point[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("TF transform lookup failed!")
            return None, None, None

    def depth_callback(self, msg):
        """ Store the latest depth frame. """
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def image_callback(self, msg):
        """ Detect weeds, refine root location using depth, and publish 3D coords. """
        if self.latest_depth is None:
            rospy.logwarn("No depth data yet, skipping detection.")
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(cv_image)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Ensure bounding box is within image bounds
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(cv_image.shape[1], x2); y2 = min(cv_image.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                # Extract ROI in the depth image
                roi_depth = self.latest_depth[y1:y2, x1:x2].copy()
                # Convert mm to m if needed
                if roi_depth.any() > 10:
                    roi_depth = roi_depth / 1000.0

                # 1) Find the maximum depth in the bounding box (assuming camera points downward)
                max_depth = np.nanmax(roi_depth)
                if np.isnan(max_depth) or max_depth <= 0:
                    rospy.logwarn("No valid depth in bounding box.")
                    continue

                # 2) Create a mask of pixels near that max depth (say, within 1 cm).
                threshold = 0.05  # 1 cm threshold
                mask = np.zeros_like(roi_depth, dtype=np.uint8)
                mask[np.abs(roi_depth - max_depth) < threshold] = 255

                # 3) Compute centroid of that mask
                M = cv2.moments(mask)
                if M["m00"] == 0:
                    rospy.logwarn("No centroid found in the root mask.")
                    continue
                centroid_u = int(M["m10"] / M["m00"])
                centroid_v = int(M["m01"] / M["m00"])

                # Convert local ROI coordinates to full image coordinates
                u = x1 + centroid_u
                v = y1 + centroid_v

                # 4) Retrieve depth at the centroid
                depth_value = self.latest_depth[v, u]
                if depth_value > 10:
                    depth_value /= 1000.0  # mm to meters
                if depth_value <= 0:
                    rospy.logwarn("Invalid depth at weed centroid.")
                    continue

                # 5) Convert to camera frame
                X_cam, Y_cam, Z_cam = self.pixel_to_world(u, v, depth_value)
                rospy.loginfo("Weed root in Camera Frame: X={:.3f}, Y={:.3f}, Z={:.3f}".format(X_cam, Y_cam, Z_cam))

                # 6) Transform to base frame
                x_base, y_base, z_base = self.transform_camera_to_base(X_cam, Y_cam, Z_cam)
                if x_base is not None:
                    rospy.loginfo("Weed root in Base Frame: X={:.3f}, Y={:.3f}, Z={:.3f}".format(x_base, y_base, z_base))

                    # Publish weed coordinates
                    weed_msg = PointStamped()
                    weed_msg.header.stamp = rospy.Time.now()
                    weed_msg.header.frame_id = "base_link"
                    weed_msg.point.x = x_base
                    weed_msg.point.y = y_base
                    weed_msg.point.z = z_base
                    self.weed_pub.publish(weed_msg)

                # Draw bounding box and centroid for visualization
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(cv_image, (u, v), 4, (0, 0, 255), -1)

                text = "X:{:.2f} Y:{:.2f} Z:{:.2f}".format(x_base, y_base, z_base)
                # Position the text above the bounding box
                cv2.putText(cv_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)



        cv2.imshow("Weed Detection", cv_image)
        cv2.waitKey(1)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    node = WeedDetectorRoot()
    node.run()
