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
from tqdm import tqdm

import matplotlib.pyplot as plt

from data import *
from model import *
from util import *

# Define Train function

# ----*********--*******------****-------***---*****----***
# -------***-----***--***----***-***-----***---***-***--***
# -------***-----******-----***---***----***---***--***-***
# -------***-----***--***--***-----****--***---***---******

def train():
    train_loader = DataLoader(Tinto(partition='train', num_points=4096, test_area=args_test_area), 
                              num_workers=2, batch_size=args_batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(Tinto(partition='test', num_points=4096, test_area=args_test_area), 
                            num_workers=2, batch_size=args_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args_cuda else "cpu")

    #Try to load models
    if args_model == 'dgcnn':
        model = DGCNN_semseg(nn.Module).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args_use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args_lr*100, momentum=args_momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args_lr, weight_decay=1e-4)

    if args_scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args_epochs, eta_min=1e-3)
    elif args_scheduler == 'step':
        scheduler = StepLR(opt, 20, 0.5, args_epochs)

    criterion = cal_loss

    best_test_iou = 0
    best_test_acc = 0
    
    plot_train = np.zeros((args_epochs,2))
    plot_test = np.zeros((args_epochs,2))

    for epoch in range(args_epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        for data, seg in train_loader:
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, args_num_class), seg.view(-1,1).squeeze())
            loss.backward()
            opt.step()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            count += batch_size
            train_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
        if args_scheduler == 'cos':
            scheduler.step()
        elif args_scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        print('Train IoU = ', train_ious)
        print('Train mean IoU = ', np.mean(train_ious))
        print('Train loss = ', train_loss*1.0/count)
        plot_train[epoch,0] = train_loss*1.0/count
        plot_train[epoch,1] = np.mean(train_ious)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch, 
                                                                                                  train_loss*1.0/count,
                                                                                                  train_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(train_ious))
        # io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        for data, seg in test_loader:
            data, seg = data.to(device), seg.to(device)
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            seg_pred = model(data)
            seg_pred = seg_pred.permute(0, 2, 1).contiguous()
            loss = criterion(seg_pred.view(-1, args_num_class), seg.view(-1,1).squeeze())
            pred = seg_pred.max(dim=2)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        print('Test IoU = ', test_ious)
        print('Test mean IoU = ', np.mean(test_ious))
        print('Test loss = ', test_loss*1.0/count)
        plot_test[epoch,0] = test_loss*1.0/count
        plot_test[epoch,1] = np.mean(test_ious)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                              test_loss*1.0/count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              np.mean(test_ious))
        # io.cprint(outstr)
        if np.mean(test_ious) >= best_test_iou:
            best_test_iou = np.mean(test_ious)
            torch.save(model.state_dict(), 'model_3DCNN/%s.t7' % (args_exp_name))
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'model_3DCNN/%s_best_acc.t7' % (args_exp_name))
        
        torch.save(model.state_dict(), 'model_3DCNN/%s-%d.t7' % (args_exp_name, epoch))
        
    return plot_train, plot_test
        
# To train !!

# Need to change:
train_path = 'lithonet_sem_seg_hdf5_data_Experiment_12'
test_path = 'lithonet_sem_seg_hdf5_data_Experiment_12'
args_exp_name = 'Experiment_12'
args_test_area = '2'
args_epochs = 120 
args_num_class = 10
args_batch_size = 32
dim_rgb = 3
dim_swir = 144
dim_geo = 28
data_dimension = 175 # RGB, SWIR (144 features), geometric (28 features)

# Arguments no need to change (following original codes):
args_cuda = True
args_model = 'dgcnn'
args_k = 20
args_emb_dims = 1024
args_dropout = 0.5
args_use_sgd = True
args_scheduler = 'cos'
args_lr = 0.001
args_momentum = 0.9

plot_train, plot_test = train() # --> All arguments need to be resolved!!

# plot_train = np.loadtxt('model/plot.txt')[:,0:2]
# plot_test = np.loadtxt('model/plot.txt')[:,2:4]

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(plot_train[:,0], 'tab:red')
axs[0, 0].set_title('Train loss')
axs[0, 1].plot(plot_test[:,0], 'tab:red')
axs[0, 1].set_title('Test loss')
axs[1, 0].plot(plot_train[:,1], 'tab:red')
axs[1, 0].set_title('Train IoU')
axs[1, 1].plot(plot_test[:,1], 'tab:red')
axs[1, 1].set_title('Test IoU')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
   ax.label_outer()
    
# plt.plot(plot_test[:,0])
# plt.show()
plt.savefig('model_3DCNN/plot.png', bbox_inches='tight')
np.savetxt('model_3DCNN/plot.txt', np.hstack((plot_train,plot_test)))
