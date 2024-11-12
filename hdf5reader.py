import h5py
import pdb
import numpy as np
from PIL import Image
import io
import cv2
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--test', type=int, default=0)
args = parser.parse_args()
 


def print_dataset_size(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, Size: {obj.size}")
        print("the first : ", obj[0])
        print("the last  : ", obj[-1])
        # if name == 'p':
        for i in range(10):
            print(obj[i])
            # print(type(obj[i]))

# with h5py.File('/home/ziyan/02_research/EventHPE/Squat_ziyan_1017_1.h5', 'r') as h5:
    
# h5 = h5py.File('/home/ziyan/02_research/EventHPE/Walking_rex_1016_1.h5', 'r')
h5 = h5py.File('/home/ziyan/02_research/EventHPE/ziyan_kick.h5', 'r')
# ['events_p', 'events_t', 'events_xy', 'image_raw_event_ts']
# f.visititems(print_dataset_size)
image_data = np.asarray(h5['images']['binary'])
image_ts = np.asarray(h5['events']['event_annot_ts'])
real_image_ts = image_ts[::8]

x = np.asarray(h5['events']['x'])
y = np.asarray(h5['events']['y'])
xy_u = np.asarray(h5['events']['xy_undistort'])
t = np.asarray(h5['events']['t'])
p = np.asarray(h5['events']['p'])
event_trigger = np.asarray(h5['events']['event_annot_ts'])

R = np.asarray(h5['annotations']['R'])
T = np.asarray(h5['annotations']['T'])
poses = np.asarray(h5['annotations']['poses'])
shape = np.asarray(h5['annotations']['shape'])

event_intr = np.asarray(h5['calibration']['event_intr'])
event_extr = np.asarray(h5['calibration']['event_extr'])
event_dist = np.asarray(h5['calibration']['event_dist'])
image_intr = np.asarray(h5['calibration']['image_intr'])
image_extr = np.asarray(h5['calibration']['image_extr'])
image_dist = np.asarray(h5['calibration']['image_dist'])

img2events = np.searchsorted(t, real_image_ts)



Rt_ei = event_extr @ np.linalg.inv(image_extr)
t_ei = Rt_ei[:3,3].reshape(-1,1)
R_ei = Rt_ei[:3,:3]
H = R_ei
# H = np.array([
#     [event_intr[1], 0., event_intr[3]],
#     [0., event_intr[2], event_intr[4]],
#     [0. ,0., 1.]
# ])@(H@np.linalg.inv(np.array([
#     [image_intr[1], 0., image_intr[3]],
#     [0., image_intr[2], image_intr[4]],
#     [0. ,0., 1.]
# ])))
# 0.475
H = np.array([
    [0.6*event_intr[1], 0., event_intr[3]],
    [0., 0.6*event_intr[2], event_intr[4]],
    [0. ,0., 1.],
    
])@(H@np.linalg.inv(np.array([
    [image_intr[1], 0., image_intr[3]],
    [0., image_intr[2], image_intr[4]],
    [0. ,0., 1.]
])))
H = H/H[-1,-1]


iTestImg = args.test
for iTestImg in range(0, len(real_image_ts)-1):
    # pdb.set_trace()
    img = Image.open(io.BytesIO(image_data[iTestImg]))
    # img = img.convert('L')  # 'L' mode in PIL is for grayscale
    img = np.array(img)

    image_warped = cv2.warpPerspective(np.array(img),H,dsize=(480,640))
    startEventIdx = img2events[iTestImg]
    endEventIdx = img2events[iTestImg+1]
    tmpx = xy_u[0][(startEventIdx):(startEventIdx+50000)]
    tmpy = xy_u[1][(startEventIdx):(startEventIdx+50000)]
    tmpP = p[(startEventIdx):(startEventIdx+50000)]
    # Draw each event as a point on the image
    for iEvent in range(len(tmpP)):
        eventY, eventX, eventP = tmpy[iEvent], tmpx[iEvent], tmpP[iEvent]
        # polarity_color = (0, 0, 255) if eventP else (255, 0, 0)
        polarity_color = [0, 0, 255] if eventP else [255, 0, 0]
        # cv2.circle(image_warped, (eventX, eventY), radius=1, color=polarity_color, thickness=-1)
        if 0 < eventY < 640 and 0 < eventX < 480:
            image_warped[int(eventY), int(eventX)] = polarity_color
    filename = "demo_images/image_with_events_{:11d}.jpg".format(real_image_ts[iTestImg])
    cv2.imwrite(filename, image_warped)
    # pdb.set_trace()
# pdb.set_trace()

h5.close()