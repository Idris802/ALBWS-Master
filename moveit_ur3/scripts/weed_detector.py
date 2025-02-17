#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import rospkg
import os

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO

import tf
import tf.transformations as tfm

class WeedDetector:
    def __init__(self):
        rospy.init_node("weed_detector", anonymous=True)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Placeholder for the camera intrinsic matrix
        self.camera_matrix = None

        # Load YOLO Model
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("moveit_ur3")
        model_path = os.path.join(package_path, "models", "best_20.pt")
        self.model = YOLO(model_path)

        # Subscribers for image, depth and camera info
        self.color_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)

        self.latest_depth = None

        # TF listener for coordinate transformations
        self.listener = tf.TransformListener()

    def camera_info_callback(self, msg):
        """
        Callback to get camera intrinsics from /camera/color/camera_info.
        """
        # Extract the intrinsic parameters from the camera info message
        fx = msg.K[0]
        fy = msg.K[4]
        cx = msg.K[2]
        cy = msg.K[5]
        self.camera_matrix = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])
        rospy.loginfo_once("Camera intrinsics received: fx: {:.2f}, fy: {:.2f}, cx: {:.2f}, cy: {:.2f}".format(fx, fy, cx, cy))

    def pixel_to_world(self, u, v, depth):
        """
        Convert pixel coordinates (u, v) and depth to real-world coordinates (X, Y, Z) using the camera intrinsic matrix.
        """
        if self.camera_matrix is None:
            rospy.logwarn("Camera intrinsics not received yet!")
            return None, None, None

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        # Compute real-world coordinates using the pinhole camera model
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth

        return X, Y, Z

    def transform_camera_to_base(self, x_cam, y_cam, z_cam):
        """
        Transform a point from the camera frame to the robot base frame using TF.
        """
        try:
            # Wait for the transform from camera_link to base_link
            self.listener.waitForTransform('/base_link', '/camera_color_optical_frame', rospy.Time(0), rospy.Duration(3.0))
            (trans, rot) = self.listener.lookupTransform('/base_link', '/camera_color_optical_frame', rospy.Time(0))

            # Convert the camera point into homogeneous coordinates
            cam_point = np.array([x_cam, y_cam, z_cam, 1])
            # Build the transformation matrix from translation and rotation (quaternion)
            transform_matrix = tfm.concatenate_matrices(tfm.translation_matrix(trans),
                                                        tfm.quaternion_matrix(rot))
            # Apply the transformation
            base_point = np.dot(transform_matrix, cam_point)
            x_base, y_base, z_base = base_point[:3]
            return x_base, y_base, z_base
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("TF transform lookup failed!")
            return None, None, None

    def depth_callback(self, msg):
        """
        Callback to store the latest depth frame.
        """
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def image_callback(self, msg):
        """
        Process the color image to detect weeds and compute their 3D position.
        """
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.model(cv_image)

        for r in results:
            for box in r.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Ensure depth data is available
                if self.latest_depth is None:
                    continue

                # Retrieve depth from the center of the bounding box
                depth_value = self.latest_depth[(y1 + y2) // 2, (x1 + x2) // 2]
                # Convert depth from millimeters to meters if needed
                if depth_value > 10:
                    depth_value = depth_value / 1000.0

                # Convert pixel coordinates to world coordinates using the dynamic camera matrix
                X, Y, Z = self.pixel_to_world((x1 + x2) // 2, (y1 + y2) // 2, depth_value)
                if X is None:
                    continue

                rospy.loginfo("Detected weed at (Camera Frame): X = {:.3f}, Y = {:.3f}, Z = {:.3f}".format(X, Y, Z))
                
                # Optionally, transform to the robot's base frame
                x_base, y_base, z_base = self.transform_camera_to_base(X, Y, Z)
                rospy.loginfo("Weed in Base Frame: X = {:.3f}, Y = {:.3f}, Z = {:.3f}".format(x_base, y_base, z_base))
                
                # Draw bounding box on the image for visualization
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                text = "X:{:.2f} Y:{:.2f} Z:{:.2f}".format(X, Y, Z)
                
                # Position the text above the bounding box
                cv2.putText(cv_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Weed Detection", cv_image)
        cv2.waitKey(1)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    detector = WeedDetector()
    detector.run()
