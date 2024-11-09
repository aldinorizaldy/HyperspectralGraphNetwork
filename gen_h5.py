# Read txt file, shift coordinates and save as numpy

import numpy as np
import os
from sklearn import preprocessing

data_folder = 'data/lithonet_data_Experiment_12'
output_folder = 'data/lithonet_sem_seg_data_Experiment_12'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

filelist = [line.rstrip() for line in open('data/lithonet_RGBSWIReig_data_label.txt')]
filepath_list = [os.path.join(data_folder, p) for p in filelist]

for filepath in filepath_list:
    elements = filepath.split('/')
    out_filename = os.path.join(output_folder,(elements[-1]).split('.')[0]+'.npy')
    
    data_label = np.load(filepath)
    xyz_min = np.amin(data_label,axis=0)[0:3]
    print(xyz_min)
    
    data_label[:,0:3] -= xyz_min
    
    # print(data_label.shape)
    np.save(out_filename, data_label)

import os
import sys
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import numpy as np
import h5py

# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, data_dtype='uint8', label_dtype='uint8'):
    h5_fout = h5py.File(h5_filename, 'w') # add 'w' to write
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.close()

import numpy as np
import glob
import os
import sys

# Constant
min_point_discard_block = 1000
# data_dimension = XYZ RGB normXYZ + SWIR
# original dimension = 9, plus SWIR dimension = 144, plus eigenfeatures 28, total = 181
data_dimension = 181

# -----------------------------------------------------------------------------
# PREPARE BLOCK DATA FOR DEEPNETS TRAINING/TESTING
# -----------------------------------------------------------------------------

def sample_data(data, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample - N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N)) + list(sample)


def sample_data_label(data, label, num_sample):
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]
    return new_data, new_label


def room2blocks(data, label, num_point, block_size=1.0, stride=1.0,
                random_sample=False, sample_num=None, sample_aug=1, use_all_points=False):
    """ Prepare block training data.
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
            assumes the data is shifted (min point is origin) and aligned
            (aligned with XYZ axis)
        label: N size uint8 numpy array from 0-12
        num_point: int, how many points to sample in each block
        block_size: float, physical size of the block in meters
        stride: float, stride for block sweeping
        random_sample: bool, if True, we will randomly sample blocks in the room
        sample_num: int, if random sample, how many blocks to sample
            [default: room area]
        sample_aug: if random sample, how much aug
    Returns:
        block_datas: K x num_point x 6 np array of XYZRGB, RGB is in [0,1]
        block_labels: K x num_point x 1 np array of uint8 labels
        
    TODO: for this version, blocking is in fixed, non-overlapping pattern.
    """
    print('start r2b')
    print('block size = ', block_size)
    print('sample num = ', sample_num)
    assert (stride <= block_size)

    limit = np.amax(data, 0)[0:3]
    print('limit = ', limit)

    # Get the corner location for our sampling blocks    
    xbeg_list = []
    ybeg_list = []
    if not random_sample:
        print('if not random sample')
        num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
        num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
        # num_block_x = int(np.ceil((limit[0] / block_size) / stride)) # Divide, not minus, no need to add 1
        # num_block_y = int(np.ceil((limit[1] / block_size) / stride)) # Divide, not minus, no need to add 1
        print('num_block_x', num_block_x)
        print('num_block_y', num_block_y)
        for i in range(num_block_x):
            # print('i = ', i)
            for j in range(num_block_y):
                # print('j = ', j)
                xbeg_list.append(i * stride)
                ybeg_list.append(j * stride)
                # xbeg_list.append(block_size * i * stride) # collect the block boundary
                # ybeg_list.append(block_size * j * stride) # collect the block boundary
        # print('xbeg_list', xbeg_list)
        # print('ybeg_list', ybeg_list)
    else:
        print('else not random sample')
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
            print('sample num if none', sample_num)
        for _ in range(sample_num):
            # print('start for ........')
            # print('_', _)
            xbeg = np.random.uniform(-block_size, limit[0])
            ybeg = np.random.uniform(-block_size, limit[1])
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)
    print('get corner done')

    # Collect blocks
    block_data_list = []
    block_label_list = []
    idx = 0
    for idx in range(len(xbeg_list)):
        xbeg = xbeg_list[idx]
        ybeg = ybeg_list[idx]
        
        # to check which data belong to which block
        xcond = (data[:, 0] <= xbeg + block_size) & (data[:, 0] >= xbeg)
        ycond = (data[:, 1] <= ybeg + block_size) & (data[:, 1] >= ybeg)
        cond = xcond & ycond
        if np.sum(cond) < min_point_discard_block:  # discard block if there are less than 20 pts (original 100 pts).
            continue

        # select data according to the boundary of the block
        block_data = data[cond, :]
        block_label = label[cond]
        # print('block data shape', block_data.shape)
        # print('block label shape', block_label.shape)

        if use_all_points:
            block_data_list.append(block_data)
            block_label_list.append(block_label)
        else:
            # randomly subsample data
            block_data_sampled, block_label_sampled = \
                sample_data_label(block_data, block_label, num_point)
            block_data_list.append(np.expand_dims(block_data_sampled, 0))
            block_label_list.append(np.expand_dims(block_label_sampled, 0))
    print('collect block done')

    if use_all_points:
        block_data_return, block_label_return = np.array(block_data_list), np.array(block_label_list)
    else:
        block_data_return, block_label_return = np.concatenate(block_data_list, 0), np.concatenate(block_label_list, 0)
    print('block_data_return_size:')
    print(np.array(block_data_return).shape)

    return block_data_return, block_label_return
    print('end r2b')


def room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                                random_sample, sample_num, sample_aug):
    """ room2block, with input filename and RGB preprocessing.
        for each block centralize XYZ, add normalized XYZ as 678 channels
    """
    print('start r2b plus normalized')
    data = data_label[:, 0:-1]
    print('data size', data.shape)
    data[:, 3:6] /= 255.0  # normalized RGB
    # SWIR has already normalized
    label = data_label[:, -1].astype(np.uint8) # The label, always in the last column!!
    max_room_x = max(data[:, 0])
    max_room_y = max(data[:, 1])
    max_room_z = max(data[:, 2])
    print('max room', max_room_x)
    print('max room', max_room_y)
    print('max room', max_room_z)

    data_batch, label_batch = room2blocks(data, label, num_point, block_size, stride,
                                          random_sample, sample_num, sample_aug)
    print('data_batch shape', data_batch.shape)
    new_data_batch = np.zeros((data_batch.shape[0], num_point, data_dimension)) # Change to number data dimension
    for b in range(data_batch.shape[0]):
        # add normalized XYZ (column 3,4,5)
        new_data_batch[b, :, 3] = data_batch[b, :, 0] / max_room_x 
        new_data_batch[b, :, 4] = data_batch[b, :, 1] / max_room_y
        new_data_batch[b, :, 5] = data_batch[b, :, 2] / max_room_z
        # recenter for each block
        minx = min(data_batch[b, :, 0])
        miny = min(data_batch[b, :, 1])
        # print('min x', minx)
        # print('min y', miny)
        # print ('before shifted', data_batch[b, 0:10, 0])
        data_batch[b, :, 0] -= (minx + block_size / 2)
        data_batch[b, :, 1] -= (miny + block_size / 2)
        # print ('after shifted', data_batch[b, 0:10, 0])
    # Colummn: XYZ, normXYZ, RGB, SWIR
    new_data_batch[:, :, 0:3] = data_batch[:,:,0:3] # for XYZ
    new_data_batch[:, :, 6:9] = data_batch[:,:,3:6] # for RGB
    new_data_batch[:,:,9:153] = data_batch[:,:,6:150] # for SWIR (144 features)
    new_data_batch[:,:,153:181] = data_batch[:,:,150:178] # for eigenfeatures (14)
    
    return new_data_batch, label_batch
    print('end r2b plus normalized')


def room2blocks_wrapper_normalized(data_label_filename, num_point, block_size=1.0, stride=1.0,
                                   random_sample=False, sample_num=None, sample_aug=1):
    print('start r2b wrapper normalized')
    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'npy':
        data_label = np.load(data_label_filename)
    else:
        print('Unknown file type! exiting.')
        exit()
    return room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                                       random_sample, sample_num, sample_aug)
    print('end r2b wrapper normalized')

import os
import numpy as np
import sys
import json

# Constants
block_size = 50
stride = 50
NUM_POINT = 4096
sample_num = 10000
random_sample = True
output_path = 'data/lithonet_sem_seg_hdf5_data_Experiment_12'

H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 181] # Change to number of data dimension # 6 XYZnormXYZ, 3 RGB, 144 SWIR, 28 eigenfeatures
label_dim = [NUM_POINT]
data_dtype = 'float32'
label_dtype = 'uint8'

# Set paths
data_dir = 'data/lithonet_sem_seg_data_Experiment_12'
filelist = 'data/lithonet_RGBSWIReig_data_list.txt'
data_label_files = [os.path.join(data_dir, line.rstrip()) for line in open(filelist)]
output_dir = output_path
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_filename_prefix = os.path.join(output_dir, 'ply_data_all')
output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
output_all_file = os.path.join(output_dir, 'all_files.txt')
fout_room = open(output_room_filelist, 'w')
all_file = open(output_all_file, 'w')

# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0 # state: the next h5 file to save

def insert_batch(data, label, last_batch=False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        buffer_size += data_size
    else: # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...] 
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...] 
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype) 
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return


sample_cnt = 0
for i, data_label_filename in enumerate(data_label_files):
    print(data_label_filename)
    # training
    # data, label = util.room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=15.0, stride=1,
    #                                                      random_sample=True, sample_num=10000)
    # testing
    print('before room2block')
    data, label = room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=block_size, stride=stride,
                                                         random_sample=random_sample, sample_num=sample_num)
    print('after room2block')
    print('{0}, {1}'.format(data.shape, label.shape))
    for _ in range(data.shape[0]):
        fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')

    sample_cnt += data.shape[0]
    insert_batch(data, label, i == len(data_label_files)-1)

fout_room.close()
print("Total samples: {0}".format(sample_cnt))

for i in range(h5_index):
    all_file.write(os.path.join(output_path[5:], 'ply_data_all_') + str(i) +'.h5\n')
all_file.close()

print('H5 done!!')