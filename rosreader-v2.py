#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

# Replace EventMsg with the correct message type for your events
from dvs_msgs.msg import EventArray  # For example, this could be an array of Event messages
import glob
import os

class DVSImageProcessor:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('dvs_image_processor', anonymous=True)
        
        # Create a CvBridge instance for converting ROS Image messages to OpenCV images
        self.bridge = CvBridge()
        
        # Set up subscribers
        rospy.Subscriber("/feature_tracks", Image, self.image_callback)
        
        # Initialize variables for storing image and events
        self.image = None
        self.events = []
        self.image_counter = 0 

    def image_callback(self, img_msg):
        # Convert ROS Image message to OpenCV format
        try:
            self.image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
            filename = "feature_tracks/image_with_events_{:04d}.jpg".format(self.image_counter)  # seconds and nanoseconds
            print("ok")
            cv2.imwrite(filename, self.image)
            rospy.loginfo("Image with events saved at {}".format(filename))
            self.image_counter +=1

        except:
            rospy.logerr("Failed to convert image message to OpenCV")

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz processing rate
        while not rospy.is_shutdown():
            print("alive")
            rate.sleep()

if __name__ == '__main__':
    folder_path = 'feature_tracks'
    image_extension = '*.jpg'  # Change this to the appropriate extension (e.g., *.png, *.jpeg)

    # Get all image files in the folder
    images = glob.glob(os.path.join(folder_path, image_extension))

    # Loop through and remove each image file
    for image in images:
        os.remove(image)
    
    try:
        processor = DVSImageProcessor()
        processor.run()
    except rospy.ROSInterruptException:
        pass
