#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header

# Replace EventMsg with the correct message type for your events
from dvs_msgs.msg import EventArray  # For example, this could be an array of Event messages
import message_filters

class DVSImageProcessor:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('dvs_image_processor', anonymous=True)
        
        # Create a CvBridge instance for converting ROS Image messages to OpenCV images
        self.bridge = CvBridge()
        
        # Set up subscribers
        image_sub = message_filters.Subscriber("/dvs/image_raw", Image)
        event_sub = message_filters.Subscriber("/dvs/events", EventArray)

        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, event_sub], queue_size=100, slop=10)

        self.ts.registerCallback(self.callback)

        # Initialize variables for storing image and events
        self.image = None
        self.events = []


    def callback(self, img_msg, event_array_msg):
        # Check if there is an image and events to process
        print("alive")
        try:
            image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logerr("Failed to convert image message to OpenCV: {}".format(e))
            return
        
        # Create a copy of the image to draw on
        image_with_events = image.copy()
        
            
        # Draw each event as a point on the image
        for event in event_array_msg.events:
            x, y = event.x, event.y
            polarity_color = (0, 0, 255) if event.polarity else (255, 0, 0)
            cv2.circle(image_with_events, (x, y), radius=2, color=polarity_color, thickness=-1)

        if event_array_msg.events:
            timestamp = event_array_msg.events[0].ts
            img_timestamp = img_msg.header.stamp
            print("events[0]: ", event_array_msg.events[0].ts, " events[-1]: ", event_array_msg.events[-1].ts,
                  " img_msg: ", img_timestamp)
            filename = "eventsOnImgs/image_with_events_{:11d}_{:09d}.jpg".format(timestamp.secs, timestamp.nsecs)
            cv2.imwrite(filename, image_with_events)
            rospy.loginfo("Image with events saved at {}".format(filename))
        

    def run(self):
        rospy.spin()  # Keep the node running until interrupted

if __name__ == '__main__':
    try:
        processor = DVSImageProcessor()
        processor.run()
    except rospy.ROSInterruptException:
        pass
