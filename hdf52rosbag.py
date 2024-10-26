import h5py
import cv2
import rospy
import rosbag
from dvs_msgs.msg import Event  
from dvs_msgs.msg import EventArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import glob
import re
import pdb
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time

def convert_to_rosbag(events_hdf5, images_folder, rosbag_file):
    rospy.init_node('hdf5_images_to_rosbag', anonymous=True)
    bridge = CvBridge()
    # Start timing
    start_time = time.time()

    # Open the ROS bag file for writing
    with rosbag.Bag(rosbag_file, 'w') as bag:
        # Load the event data from HDF5
        with h5py.File(events_hdf5, 'r') as hdf5:
            print(hdf5.keys())
            event_ps = hdf5['p']
            event_xs = hdf5['x']
            event_ys = hdf5['y']
            event_ts = hdf5['t']
            image_ts = np.array(hdf5['image_raw_event_ts'])
            img2events = np.searchsorted(event_ts, image_ts)
            offset_time = image_ts[69]

            if (1):
                # Iterate through events with tqdm for progress tracking
                print("Converting events to eventArray...")
                # for i in tqdm(range(len(img2events) - 1), desc="Img2Events"):
                for i in tqdm(range(69, 77), desc="Img2Events"):
                    print("i: ", i)
                    eventArray_msg = EventArray()
                    eventArray_msg.height = 256
                    eventArray_msg.width = 256

                    start_idx = img2events[i]
                    end_idx = img2events[i+1]
                    # Create a dvs_event message
                    selected_xs = event_xs[start_idx:end_idx]   # x-coordinate
                    selected_ys = event_ys[start_idx:end_idx]   # y-coordinate
                    selected_ts = event_ts[start_idx:end_idx]   # timestamp
                    print(selected_ts)
                    selected_ps = event_ps[start_idx:end_idx]  # polarity
                    events = [Event(
                        x=int(selected_xs[j]),
                        y=int(selected_ys[j]),
                        ts=rospy.Time(secs=int((selected_ts[j]-offset_time+1) // 1000000),  # seconds
                        nsecs=int(((selected_ts[j]-offset_time+1) % 1000000) * 1000)),  # nanoseconds
                        polarity=bool(selected_ps[j])
                    ) for j in range(end_idx - start_idx)]
                    eventArray_msg.events = events
                    # Write the event message to the ROS bag
                    print(events[0].ts)
                    bag.write('/dvs/events', eventArray_msg, events[0].ts)
            
        # Iterate through the images in the folder and write them to the bag
        if (1):
            print("Converting images...")
            image_files = sorted(glob.glob(os.path.join(str(images_folder), "*.jpg")))[70:78]
            for img_file in tqdm(image_files, desc="Images"):
                # Load the image
                img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

                # Convert the image to a ROS Image message
                img_msg = bridge.cv2_to_imgmsg(img, encoding="mono8")

                # Set the timestamp (using the filename as a proxy for timestamp, assuming sequential naming)
                # Extract the file name without the extension
                file_name = os.path.splitext(os.path.basename(img_file))[0]
                # Extract the number from the file name using string slicing or regex
                number_str = ''.join(filter(str.isdigit, file_name))
                
                # Convert the number string to integer
                timestamp = image_ts[int(number_str)] - offset_time +1
                image_time=rospy.Time(secs=int(timestamp // 1000000),  # seconds
                    nsecs=int((timestamp % 1000000) * 1000))  # nanoseconds
                
                # Write the image message to the ROS bag
                print(image_time)
                bag.write('/dvs/image_raw', img_msg, image_time)
    # Record total execution time
    end_time = time.time()
    total_time = end_time - start_time
    print("Conversion completed in ", total_time, " seconds")

if __name__ == "__main__":
    # events_hdf5_path = '/home/ziyan/02_research/EventHPE/h5py/subject01_group1_time1/events.hdf5'
    # images_folder_path = '/home/ziyan/02_research/EventHPE/data_event/data_event_out/full_pic_256/subject01_group1_time1/'
    events_hdf5_path = Path('./events.hdf5')
    images_folder_path = Path('./data_event/data_event_out/full_pic_256/subject01_group1_time1')
    
    # rosbag_output_path = 'subject01_group1_time1.bag'
    rosbag_output_path = 'tmp_seq1.bag'
    
    convert_to_rosbag(events_hdf5_path, images_folder_path, rosbag_output_path)
