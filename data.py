# Define data

import os
import sys
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

def load_data_semseg(partition, test_area):
    DATA_DIR = 'data'
    #prepare_test_data_semseg()
    if partition == 'train':
        data_dir = os.path.join(DATA_DIR, train_path)
    else:
        data_dir = os.path.join(DATA_DIR, test_path)
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg

class Tinto(Dataset):
    def __init__(self, num_points=4096, partition='train', test_area='1'):
        self.data, self.seg = load_data_semseg(partition, test_area)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]

class TintoDataset_eval(Dataset):  # load data block by block, without using h5 files
    def __init__(self, split='train', data_root='trainval_fullarea', num_point=4096, test_area='5', block_size=1.0, stride=1.0, num_class=20, use_all_points=False, num_thre = 1024):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.use_all_points = use_all_points
        self.stride = stride
        self.num_thre = num_thre
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []

        room_idxs = []
        for index, room_name in enumerate(rooms_split):
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path)
            points, labels = room_data[:, 0:-1], room_data[:, -1] # CHECK !! label always in the last column !!
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            block_points, block_labels = room2blocks(points, labels, self.num_point, block_size=self.block_size,
                                                       stride=self.stride, random_sample=False, sample_num=None, use_all_points=self.use_all_points)
            room_idxs.extend([index] * int(block_points.shape[0]))  # extend with number of blocks in a room
            self.room_points.append(block_points), self.room_labels.append(block_labels)
        self.room_points = np.concatenate(self.room_points)
        self.room_labels = np.concatenate(self.room_labels)

        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):  # get items in one block
        room_idx = self.room_idxs[idx]
        selected_points = self.room_points[idx]   # num_point * XYZ RGB SWIR geo
        current_labels = self.room_labels[idx]   # num_point
        center = np.mean(selected_points, axis=0)
        N_points = selected_points.shape[0]

        current_points = np.zeros((N_points, data_dimension+6))  # data dimension + XYZ + normXYZ, Check!
        # add normalized XYZ (column 3,4,5)
        current_points[:, 3] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 4] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 5] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        # recenter for each block
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        # normalize RGB
        selected_points[:, 3:6] /= 255.0
        # SWIR already normalized
        # Colummn: XYZ, normXYZ, RGB, SWIR, geo
        current_points[:,0:3] = selected_points[:,0:3] # for XYZ
        current_points[:,6:9] = selected_points[:,3:6] # for RGB
        current_points[:,9:153] = selected_points[:,6:150] # for SWIR
        current_points[:,153:181] = selected_points[:,150:178] # for geo

        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)
