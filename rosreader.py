#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image

def image_callback(msg):
    # Extract the timestamp from the header
    timestamp = msg.header.stamp
    # Print the seconds and nanoseconds part of the timestamp
    rospy.loginfo("Time: secs=%d, nsecs=%d", timestamp.secs, timestamp.nsecs)

def image_time_printer():
    # Initialize the ROS node
    rospy.init_node('image_time_printer', anonymous=True)
    
    # Subscribe to the /dvs/image_raw topic
    rospy.Subscriber("/dvs/image_raw", Image, image_callback)
    
    # Keep the node running
    rospy.spin()

if __name__ == '__main__':
    try:
        image_time_printer()
    except rospy.ROSInterruptException:
        pass
