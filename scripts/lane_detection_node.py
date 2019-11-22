#!/usr/bin/env python

import rospy
import cv2
import numpy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from LaneDetector

class lane_detector:
    def __init__(self):
        self.bridge = CvBridge()
        self.Detector = LaneDetector()
        # TODO: Configure via json
        self.image_sub = rospy.Subscriber("camera/image_raw", Image, self.callback)
        # publish coefficients of spline as array
        self.spline_pub = rospy.Publisher("lane_splines", Float32MultiArray)
        # publish top down view for visualization
        self.image_pub = rospy.Publisher("top_down", Image)


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
        cv_img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        warped_image = Detector.perspective_warp(cv_img)
        filtered_image = Detector.filter(warped_image)
        splines = Detector.fit_splines(filtered_image)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(filtered_image, 'mono8'))
            self.spline_pub.publish(self.float_msg)


def main(args):
    det = lane_detector()
    rospy.init_node('lane_detector', anonymous = True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting Down"

if __name__ == '__main__':
    main(sys.argv)

