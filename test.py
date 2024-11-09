from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import *
from model import *
from util import *

def test():
    DUMP_DIR = path_dump_dir
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []

    dataset = TintoDataset_eval(split='test', data_root=args_data_dir, num_point=args_num_points, test_area=args_test_area,
                           block_size=args_block_size, stride=args_block_size, num_class=args_num_classes, num_thre=100, use_all_points=True)
    test_loader = DataLoader(dataset, batch_size=args_test_batch_size, shuffle=False, drop_last=False)

    room_idx = np.array(dataset.room_idxs)
    num_blocks = len(room_idx)

    fout_data_label = []
    for room_id in np.unique(room_idx):
        out_data_label_filename = 'Area_%s_pred_gt_%s.txt' % (test_area, args_predict_name)
        out_data_label_filename = os.path.join(DUMP_DIR, out_data_label_filename)
        fout_data_label.append(open(out_data_label_filename, 'w+'))     

    device = torch.device("cuda" if args_cuda else "cpu")

    # io.cprint('Start overall evaluation...')

    # Try to load models
    if args_model == 'dgcnn':
        model = DGCNN_semseg(nn.Module).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(path_model))
    model = model.eval()

    print('model restored')

    test_acc = 0.0
    count = 0.0
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []

    print('Start testing ...')
    num_batch = 0
    for data, seg in tqdm(test_loader):
        th_subblock = 20000
        st_sb = 0
        en_sb = th_subblock
        if data.shape[1] > th_subblock:
            print('Too many points in the block. Split the block!!')
            
            # Split data into n sub-blocks
            n_subblocks = int(np.ceil(data.shape[1]/th_subblock))
            print('N subblocks', n_subblocks)
            for split in range(n_subblocks):
                print('Working on subblock = ', split+1)
                if split+1 < n_subblocks:
                    n_pts_subblock = th_subblock
                else:
                    n_pts_subblock = data.shape[1] - (split*th_subblock) 
                
                data_split = torch.zeros([1,n_pts_subblock,data.shape[2]],dtype=torch.float64)
                seg_split = torch.zeros([1,n_pts_subblock],dtype=torch.float64)
                data_split[:,:,:] = data[:,st_sb:en_sb,:]
                seg_split[:,:] = seg[:,st_sb:en_sb]
                data_split, seg = data_split.to(device), seg.to(device)
                data_split = data_split.permute(0, 2, 1).float()
                batch_size = data_split.size()[0]
                
                st_sb += th_subblock
                en_sb += th_subblock

                seg_pred = model(data_split)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                pred = seg_pred.max(dim=2)[1]
                seg_np = seg_split.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)

                # write prediction results

                for batch_id in range(batch_size):
                    pts = data_split[batch_id, :, :]
                    pts = pts.permute(1, 0).float()
                    l = seg_split[batch_id, :]
                    pts[:, 6:9] *= 255.0 # unnormalized RGB, previously in 3:6
                    pred_ = pred[batch_id, :]
                    logits = seg_pred[batch_id, :, :]
                    # compute room_id
                    room_id = room_idx[num_batch + batch_id]
                    for i in range(pts.shape[0]):
                        fout_data_label[room_id].write('%f %f %f %d %d %d %d %d\n' % (
                           # change the position of normXYZ from 6,7,8 to 3,4,5
                           pts[i, 3]*dataset.room_coord_max[room_id][0], pts[i, 4]*dataset.room_coord_max[room_id][1], pts[i, 5]*dataset.room_coord_max[room_id][2],
                           pts[i, 6], pts[i, 7], pts[i, 8], pred_[i], l[i]))  # xyzRGB pred gt
                

        else:
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1).float()
            batch_size = data.size()[0]

            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            pred = seg_pred.max(dim=2)[1]
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)

            # write prediction results

            for batch_id in range(batch_size):
                pts = data[batch_id, :, :]
                pts = pts.permute(1, 0).float()
                l = seg[batch_id, :]
                pts[:, 6:9] *= 255.0 # unnormalized RGB, previously in 3:6
                pred_ = pred[batch_id, :]
                logits = seg_pred[batch_id, :, :]
                # compute room_id
                room_id = room_idx[num_batch + batch_id]
                for i in range(pts.shape[0]):
                    fout_data_label[room_id].write('%f %f %f %d %d %d %d %d\n' % (
                       # change the position of normXYZ from 6,7,8 to 3,4,5
                       pts[i, 3]*dataset.room_coord_max[room_id][0], pts[i, 4]*dataset.room_coord_max[room_id][1], pts[i, 5]*dataset.room_coord_max[room_id][2],
                       pts[i, 6], pts[i, 7], pts[i, 8], pred_[i], l[i]))  # xyzRGB pred gt
            
        num_batch += batch_size
        torch.cuda.empty_cache()

    for room_id in np.unique(room_idx):
        fout_data_label[room_id].close()

    # test_ious = calculate_sem_IoU(test_pred_cls, test_true_cls, args_num_classes)
    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    # test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    # avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    # test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    # outstr = 'Test :: test area: %s, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_area,
    #                                                                                        test_acc,
    #                                                                                        avg_per_class_acc,
    #                                                                                        np.mean(test_ious))
    # io.cprint(outstr)

    # calculate confusion matrix
    conf_mat = metrics.confusion_matrix(test_true_cls, test_pred_cls)
    print('Confusion matrix:')
    print(conf_mat)
    np.savetxt('predict_3DCNN/con_mat.txt', conf_mat)
    
    # calculate overall accuracy
    OA = metrics.accuracy_score(test_true_cls, test_pred_cls)
    print('Overall Accuracy')
    print(OA)
    # np.savetxt('predict/OA.txt', OA)
    # io.cprint(str(conf_mat))

    # all_true_cls.append(test_true_cls)
    # all_pred_cls.append(test_pred_cls)
    # all_true_seg.append(test_true_seg)
    # all_pred_seg.append(test_pred_seg)'''

args_data_dir = 'data/lithonet_sem_seg_data_Experiment_12' # CHANGE
args_num_points = 4096
args_test_area = '2'
test_area = 2
args_block_size = 50 # CHANGE # CUDA out of memory for 100 m block size
args_num_classes = 10 # CHANGE
args_test_batch_size = 1
args_cuda = True
args_model = 'dgcnn'
args_k = 20
args_emb_dims = 1024
args_dropout = 0.5
args_num_class = 10
args_predict_name = 'Experiment_12'

data_dimension = 175 # RGB (3) SWIR (144) geo (28)
dim_rgb = 3
dim_swir = 144
dim_geo = 28

path_dump_dir = 'predict_3DCNN'
path_model = 'model_3DCNN/Experiment_12_best_acc.t7'

test()