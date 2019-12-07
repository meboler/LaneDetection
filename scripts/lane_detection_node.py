#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import sys

from sensor_msgs.msg import Image
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
        R = np.asarray([[0, -1, 0],
                        [0, 0, -1],
                        [1, 0, 0]], np.float32)
        
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
        self.filter_pub = rospy.Publisher("lane_detector/filtered", Image)


        float_msg = Float32MultiArray()
        float_msg.layout.dim.append(MultiArrayDimension())
        float_msg.layout.dim.append(MultiArrayDimension())
        float_msg.layout.dim[0].label = "spline"
        float_msg.layout.dim[1].label = "coefficient"
        float_msg.layout.dim[0].size = 2
        float_msg.layout.dim[1].size = 10
        float_msg.layout.dim[0].stride = 2*10
        float_msg.layout.dim[1].stride = 10
        float_msg.layout.data_offset = 0
        float_msg.data = [0]*2*10

        self.float_msg = float_msg

    def callback(self, data):
        cv_img = self.bridge.imgmsg_to_cv2(data, "rgb8")
        warped_image = self.Detector.perspective_warp(cv_img)
        filtered_image = self.Detector.filter(warped_image)
        (left, center, right) = self.Detector.sliding_window(filtered_image)
        lane_image = self.Detector.draw_lanes(cv_img, left, right)
        self.visualization_pub.publish(self.bridge.cv2_to_imgmsg(lane_image, 'rgb8'))
        self.filter_pub.publish(self.bridge.cv2_to_imgmsg(filtered_image*255, 'mono8'))

def main(args):
    det = lane_detector()
    rospy.init_node('lane_detector', anonymous = True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting Down"

if __name__ == '__main__':
    main(sys.argv)

