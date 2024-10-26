import h5py
import pdb
import numpy as np

def print_dataset_size(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}, Size: {obj.size}")
        print("the first : ", obj[0])
        print("the last  : ", obj[-1])
        # if name == 'p':
        for i in range(10):
            print(obj[i])
            # print(type(obj[i]))

with h5py.File('/home/ziyan/02_research/EventHPE/events.hdf5', 'r') as f:
    # ['events_p', 'events_t', 'events_xy', 'image_raw_event_ts']
    f.visititems(print_dataset_size)