#!/usr/bin/env python

import rospy 
import cv2
import numpy as np
import sys

from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge

from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from LaneDetector.LaneDetector import *

class lane_detector:
    def __init__(self):
        self.bridge = CvBridge() 
        """ Set up all needed params """
        height = 480
        width = 640

        h = 1.518 # meters
        t = np.asarray([0, 0, -h], np.float32)
        # Map from world frame to camera frame
        pitch = np.deg2rad(5) # positive tilts down
        R = np.asarray([[0, -1, 0],
                        [np.sin(pitch), 0, -1*np.cos(pitch)],
                        [1*np.cos(pitch), 0, np.sin(pitch)]], np.float32)
        
        K = np.asarray([[617.2716, 0, 327.2818],
                        [0, 617.1263, 245.0939],
                        [0, 0, 1]], np.float32)
        
        D = np.asarray([0, 0, 0, 0, 0], np.float32)
        FOV_h = np.radians(91.2) 
        FOV_v = np.radians(65.5) 
        params = CameraParams()
        params.K = K
        params.D = D
        params.FOV_h = FOV_h
        params.FOV_v = FOV_v
        params.height = height
        params.width = width
        self.Detector = LaneDetector(R, t, params)
        # TODO: Configure via json
        self.image_sub = rospy.Subscriber("camera/color/image_raw", Image, self.callback)
        # publish coefficients of spline as array
        #self.spline_pub = rospy.Publisher("lane_splines", Float32MultiArray)
        # publish top down view for visualization
        self.visualization_pub = rospy.Publisher("lane_detector/visualization", Image)
        self.mask_pub = rospy.Publisher("lane_detector/mask", Image)
        self.hsv_pub = rospy.Publisher("lane_detector/color_threshold", Image)
        self.grad_pub = rospy.Publisher("lane_detector/gradient_threshold", Image)


        self.nav_pub = rospy.Publisher("lane_detector/waypoints", Path) 
    def callback(self, data):
        cv_img = self.bridge.imgmsg_to_cv2(data, "rgb8")
        # Run pipeline

        hsv_mask = self.Detector.color_threshold(cv_img)
        grad_mask = self.Detector.gradient_threshold(cv_img)

        mask_img = cv.bitwise_or(hsv_mask, grad_mask)

        blur_img = self.Detector.blur_mask(mask_img)
        warped_image = self.Detector.perspective_warp(blur_img)
        try:
            (left, center, right) = self.Detector.sliding_window(warped_image)
            waypoints = self.Detector.generate_waypoints(cv_img, center)
            # Generate publishing stuff
            lane_image = self.Detector.draw_lanes(cv_img, left, right)
            path = Path()
            path.header = data.header
            num_points = waypoints.shape[1]
            for i in range(num_points):
                x = float(waypoints[0,i])
                y = float(waypoints[1,i])
                theta = waypoints[2,i]
                w = np.cos(theta/2)
                z = np.sin(theta/2)
                pose = PoseStamped()
                p = Pose()
                p.position.x = x
                p.position.y = y
                p.position.z = 0
                p.orientation.x = 0.0
                p.orientation.y = 0.0
                p.orientation.z = z
                p.orientation.w = w
                pose.pose = p
		        pose.header = data.header
                path.poses.append(pose)
	    
            self.nav_pub.publish(path)
            self.visualization_pub.publish(self.bridge.cv2_to_imgmsg(lane_image, 'rgb8'))
        except:
            print("Failed to generate path")
	    rospy.logerr("LOLNO")
        # Publish messages
        self.hsv_pub.publish(self.bridge.cv2_to_imgmsg(hsv_mask, 'mono8'))
        self.grad_pub.publish(self.bridge.cv2_to_imgmsg(grad_mask, 'mono8'))

def main(args):
    det = lane_detector()
    rospy.init_node('lane_detector', anonymous = True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting Down"

if __name__ == '__main__':
    main(sys.argv)

