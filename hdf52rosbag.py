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
import io
from PIL import Image

def convert_to_rosbag(events_hdf5, images_folder, rosbag_file):
    rospy.init_node('hdf5_images_to_rosbag', anonymous=True)
    bridge = CvBridge()
    # Start timing
    start_time = time.time()

    # Open the ROS bag file for writing
    with rosbag.Bag(rosbag_file, 'w') as bag:
        # Load the event data from HDF5
        with h5py.File(events_hdf5, 'r') as h5:
            print(h5.keys())

            image_data = np.asarray(h5['images']['binary'])
            image_ts = np.asarray(h5['images']['image_annot_ts'])

            event_xs = np.asarray(h5['events']['x'])
            event_ys = np.asarray(h5['events']['y'])
            event_ts = np.asarray(h5['events']['t'])
            event_ps = np.asarray(h5['events']['p'])

            event_intr = np.asarray(h5['calibration']['event_intr'])
            event_extr = np.asarray(h5['calibration']['event_extr'])
            event_dist = np.asarray(h5['calibration']['event_dist'])
            image_intr = np.asarray(h5['calibration']['image_intr'])
            image_extr = np.asarray(h5['calibration']['image_extr'])
            image_dist = np.asarray(h5['calibration']['image_dist'])

            Rt_ei = np.matmul(event_extr, np.linalg.inv(image_extr))
            t_ei = Rt_ei[:3,3].reshape(-1,1)
            R_ei = Rt_ei[:3,:3]
            H = R_ei
            H = np.matmul(np.array([
                [0.475*event_intr[1], 0., event_intr[3]+2],
                [0., 0.475*event_intr[2], event_intr[4]],
                [0. ,0., 1.]]), np.matmul(H, np.linalg.inv(np.array([
                [image_intr[1], 0., image_intr[3]],
                [0., image_intr[2], image_intr[4]],
                [0. ,0., 1.]]))))
            H = H/H[-1,-1]


            
            real_image_ts = image_ts[::8]
            img2events = np.searchsorted(event_ts, real_image_ts)
            # offset_time = image_ts[0]
            # Initialize sequence counter for header
            seq_counter = 0

            if (1):
                # Iterate through events with tqdm for progress tracking
                print("Converting events to eventArray...")
                # for i in tqdm(range(len(img2events) - 1), desc="Img2Events"):
                for i in tqdm(range(0, len(real_image_ts)-1), desc="Img2Events"):
                    print("i: ", i)
                    eventArray_msg = EventArray()
                    eventArray_msg.height = 640
                    eventArray_msg.width = 480

                    start_idx = img2events[i]
                    end_idx = img2events[i+1]
                    # Create a dvs_event message
                    selected_xs = event_xs[start_idx:end_idx]   # x-coordinate
                    selected_ys = event_ys[start_idx:end_idx]   # y-coordinate
                    selected_ts = event_ts[start_idx:end_idx]   # timestamp
                    # print(selected_ts[:10], selected_ts[-10:])
                    selected_ps = event_ps[start_idx:end_idx]  # polarity
                    events = [Event(
                        x=int(selected_ys[j]),
                        y=int(selected_xs[j]),
                        ts=rospy.Time(secs=int((selected_ts[j]) // 1000000),  # seconds
                        nsecs=int(((selected_ts[j]) % 1000000) * 1000)),  # nanoseconds
                        polarity=bool(selected_ps[j])
                    ) for j in range(end_idx - start_idx)]
                    eventArray_msg.events = events
                    # Write the event message to the ROS bag
                     # Populate the header for EventArray
                    # eventArray_msg.header.seq = seq_counter
                    # eventArray_msg.header.stamp = events[-1].ts  # Set to the timestamp of the last event
                    # eventArray_msg.header.frame_id = "dvs_frame"

                    # Increment the sequence counter for the next message
                    seq_counter += 1


                    timestamp = real_image_ts[i]

                    print("events[0]: ", events[0].ts, " events[-1]: ", events[-1].ts, " timestamp: ", timestamp)
                    image_time=rospy.Time(secs=int(timestamp // 1000000),  # seconds
                        nsecs=int((timestamp % 1000000) * 1000))  # nanoseconds
                    # Convert the image to a ROS Image message
                    # Convert to grayscale
                    img = Image.open(io.BytesIO(image_data[i]))
                    img = img.convert('L')  # 'L' mode in PIL is for grayscale
                    img = np.array(img)
                    image_warped = cv2.warpPerspective(np.array(img),H,dsize=(480,640))
                        

                    img_msg = bridge.cv2_to_imgmsg(image_warped, encoding="mono8")

                    selected_xs
                    selected_ys
                    image_warped[selected_xs, selected_ys] = [255]
                    filename = "eventAndImageInBag/image_with_events_{:10d}.jpg".format(timestamp)  # seconds and nanoseconds
                    cv2.imwrite(filename, image_warped)
                    bag.write('/dvs/events', eventArray_msg, events[-1].ts)
                    img_msg.header.stamp = image_time
                    bag.write('/dvs/image_raw', img_msg, image_time)
                    
            
        # Iterate through the images in the folder and write them to the bag
        if (0):
            print("Converting images...")
            image_files = sorted(glob.glob(os.path.join(str(images_folder), "*.jpg")))[69:100]
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
    events_hdf5_path = Path('./Squat_ziyan_1017_1.h5')
    images_folder_path = Path('./data_event/data_event_out/full_pic_256/subject01_group1_time1')
    
    # rosbag_output_path = 'subject01_group1_time1.bag'
    rosbag_output_path = 'tmp_seq1.bag'
    
    convert_to_rosbag(events_hdf5_path, images_folder_path, rosbag_output_path)
