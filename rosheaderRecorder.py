#!/usr/bin/env python

import rospy
from dvs_msgs.msg import EventArray  # Adjust to the correct message type if different
from std_msgs.msg import Header

class EventHeaderRecorder:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('event_header_recorder', anonymous=True)
        
        # Set up the subscriber to the /dvs/events topic
        rospy.Subscriber("/dvs/events", EventArray, self.event_callback)
        
        # Open the file to save headers
        self.file = open("event_headers.txt", "a")  # Append mode to avoid overwriting on each run

    def event_callback(self, msg):
        # Extract header information
        header = msg.header
        seq = header.seq
        timestamp = header.stamp
        frame_id = header.frame_id

        # Format the header data as a string
        header_info = "seq: {:02d}, stamp: {:03d}.{:10d}, frame_id: {:s}\n".format(
            seq, timestamp.secs, timestamp.nsecs, frame_id
        )
        
        # Write header information to file
        self.file.write(header_info)
        rospy.loginfo("Recorded header: {}".format(header_info))

    def run(self):
        rospy.spin()  # Keep the node running and processing callbacks

    def __del__(self):
        # Close the file when the node is terminated
        self.file.close()

if __name__ == '__main__':
    try:
        recorder = EventHeaderRecorder()
        recorder.run()
    except rospy.ROSInterruptException:
        pass
