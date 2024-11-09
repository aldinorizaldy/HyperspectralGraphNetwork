import numpy as np
import os
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix

scene_path = 'data/lithonet_data_Experiment_12'
scene_file = 'Area_2_XYZRGBSWIRgeo.npy'

pred_path = 'predict_3DCNN'
pred_file = 'Area_2_pred_gt_Experiment_12.txt' 

output_file = 'Area_2_pred_gt_scene_Experiment_12.txt'

pc_orig = np.load(os.path.join(scene_path, scene_file))
xyz_min = np.amin(pc_orig, axis=0)[0:3]
print(xyz_min)
print('Make sure that the xyz_min is the same as the xyz_min during the data preprocessing!!')

pc_pred = np.loadtxt(os.path.join(pred_path, pred_file))
pc_pred_scene = pc_pred
pc_pred_scene[:,0:3] += xyz_min

np.savetxt(os.path.join(pred_path,output_file), pc_pred_scene[:,np.r_[0:3,6:8]], fmt='%f %f %f %d %d')