import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import cv2
from tensorboardX import SummaryWriter
import sys
sys.path.append('../')
# from event_pose_estimation.model import EventTrackNet
from event_pose_estimation.dataloader import TrackingDataloader
# from event_pose_estimation.loss_funcs import compute_losses, compute_mpjpe, compute_pa_mpjpe, compute_pelvis_mpjpe
# import collections
import numpy as np
from event_pose_estimation.SMPL import SMPL, batch_rodrigues
import joblib
import pdb
import pickle
# from scipy.spatial import KDTree
from eventcap_util import findCloestPoint, findClosestPointTorch, \
    find_closest_events, findBoundaryPixels, joints2dOnImage, \
    joints2dandFeatOnImage, draw_feature_dots, draw_skeleton, findImgFeat, evaluation
from event_pose_estimation.geometry import projection_torch, rot6d_to_rotmat, delta_rotmat_to_rotmat
from event_pose_estimation.loss_funcs import compute_mpjpe, compute_pa_mpjpe, compute_pa_mpjpe_eventcap, \
                                            compute_pelvis_mpjpe, compute_pck_head
import h5py
import random

def test(args):
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")

    smpl_dir = args.smpl_dir
    print('[smpl_dir] %s' % smpl_dir)

    action = args.action
    sequence_length = 1345
    numImgsInSplit = 8
    scale = args.img_size / 1280.
    cam_intr = np.array([1679.3, 1679.3, 641, 641]) * scale
    cam_intr = torch.from_numpy(cam_intr).float()
    H, W = args.img_size, args.img_size

    # set model
    modelSeq = SMPL(smpl_dir, sequence_length)
    # Freeze the rest of the SMPL model parameters
    for param in modelSeq.parameters():
        param.requires_grad = False  # Freeze all SMPL parameters

    modelSeq = modelSeq.to(device=device)  # move the model parameters to CPU/GPU

    print('===== HMR Parameters and Results =====')
    joints2DHMR_list, joints3DHMR_list, transHMR_list = [], [], []
    paramsHMR_list = []
    for iImg in range(sequence_length):
        joints2DHMR, joints3DHMR, paramsHMR, transHMR, _ = \
                joblib.load('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (args.data_dir, action, iImg))
        if iImg == 0:
            tmpJoints2DHMR = joints2DHMR
            tmpJoints3DHMR = joints3DHMR
            tmpTransHMR = transHMR
        joints2DHMR_list.append(joints2DHMR)
        joints3DHMR_list.append(joints3DHMR+np.tile(transHMR,(joints3DHMR.shape[0],1)))
        transHMR_list.append(transHMR)
        paramsHMR_list.append(paramsHMR)
    joints2DHMRForEveryFrame = torch.from_numpy(np.stack(joints2DHMR_list)).to(device)
    joints3DHMRForEveryFrame = torch.from_numpy(np.stack(joints3DHMR_list)).to(device)
    transHMRForEveryFrame = torch.from_numpy(np.stack(transHMR_list)).to(device)
    paramsHMRForEveryFrame = torch.from_numpy(np.stack(paramsHMR_list)).to(device)
    # Convert grayscale image to BGR format for color drawing
    grey_img = cv2.imread('/home/ziyan/02_research/EventHPE/data_event/data_event_out/full_pic_256/subject03_group1_time1/fullpic0000.jpg', cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    output_image = cv2.cvtColor(grey_img, cv2.COLOR_GRAY2BGR)

    # Draw dots for each feature in the data array
    for iJoint in range(tmpJoints2DHMR.shape[0]):
        x = int(tmpJoints2DHMR[iJoint][0])                # x coordinate
        y = int(tmpJoints2DHMR[iJoint][1])                # y coordinate
        color = [255,0,0]
        # Draw a dot on the output image
        cv2.circle(output_image, (x, y), radius=1, color=color, thickness=-1)
        cv2.imwrite('tmp.jpg', output_image)
    
    
    rotmats = batch_rodrigues(paramsHMRForEveryFrame[:, 3:75].reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
    hmrVerts, hmrJoints3d, _ = modelSeq(beta=paramsHMRForEveryFrame[:,75:85].view(-1, 10),
                            theta=None,
                            get_skin=True,
                            rotmats=rotmats.view(-1, 24, 3, 3))
    hmrJoints3d_trans = hmrJoints3d
    hmrJoints2d = projection_torch(hmrJoints3d_trans, cam_intr, H, W).clone().detach()

    print('===== Optimization Parameters and Results =====')
    optimizationParameters = torch.load('learnable_parameters-hpegood.pt')
    rotmats = batch_rodrigues(optimizationParameters[:, 3:75].reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
    predVerts, predJoints3d, _ = modelSeq(beta=optimizationParameters[:,75:85].view(-1, 10),
                            theta=None,
                            get_skin=True,
                            rotmats=rotmats.view(-1, 24, 3, 3))
    predJoints3d_trans = predJoints3d + optimizationParameters[:, :3].unsqueeze(1).repeat(1, predJoints3d.shape[1], 1)
    predJoints2d = projection_torch(predJoints3d_trans, cam_intr, H, W).clone().detach()

    print('===== GT Parameters and Results =====')

    joints3DGTForEveryFrame = torch.zeros((sequence_length, 24, 3))
    paramsGTForEveryFrame = torch.zeros((sequence_length, 85))

    for iImg in range(sequence_length):
        gtBeta, gtTheta, gtTran, gtJoints3d, gtJoints2d = \
            joblib.load('%s/pose_events/%s/pose%04i.pkl' % (args.data_dir, action, iImg))
        params = np.concatenate([gtTran, gtTheta, gtBeta],axis=1)
        paramsGTForEveryFrame[iImg, :] = torch.from_numpy(params)
        joints3DGTForEveryFrame[iImg, :,:] = torch.from_numpy(gtJoints3d)
    joints3DGTForEveryFrame = joints3DGTForEveryFrame.to(device)
    paramsGTForEveryFrame = paramsGTForEveryFrame.to(device)

    print('===== Compute JPE =====')

    startSplit = args.startWindow
    endSplit = args.endWindow
    startIdx = startSplit*numImgsInSplit
    endIdx = endSplit*numImgsInSplit
    predJoints3d_trans.device
    pred_mpjpe = torch.mean(compute_mpjpe(predJoints3d_trans[startIdx: endIdx].unsqueeze(0), \
                                          joints3DGTForEveryFrame[startIdx: endIdx].unsqueeze(0)),dim=2)[0]  # [1, T, 24]
    pred_pa_mpjpe = torch.mean(compute_pa_mpjpe_eventcap(predJoints3d_trans[startIdx: endIdx], \
                                                         joints3DGTForEveryFrame[startIdx: endIdx]), dim=1)  # [T, 24]
    pred_pelvis_mpjpe = torch.mean(compute_pelvis_mpjpe(predJoints3d_trans[startIdx: endIdx].unsqueeze(0), \
                                                        joints3DGTForEveryFrame[startIdx: endIdx].unsqueeze(0)), dim=2)[0]
    pred_pckh05 = torch.sum(compute_pck_head(predJoints3d_trans[startIdx: endIdx].unsqueeze(0), \
                                             joints3DGTForEveryFrame[startIdx: endIdx].unsqueeze(0)), dim=2)[0]/24

    
    hmr_mpjpe = torch.mean(compute_mpjpe(joints3DHMRForEveryFrame[startIdx: endIdx].unsqueeze(0),\
                                          joints3DGTForEveryFrame[startIdx: endIdx].unsqueeze(0)),dim=2)[0]  # [1, T, 24]
    hmr_pa_mpjpe = torch.mean(compute_pa_mpjpe_eventcap(joints3DHMRForEveryFrame[startIdx: endIdx], \
                                                        joints3DGTForEveryFrame[startIdx: endIdx]), dim=1)  # [T, 24]
    hmr_pelvis_mpjpe = torch.mean(compute_pelvis_mpjpe(joints3DHMRForEveryFrame[startIdx: endIdx].unsqueeze(0), \
                                                        joints3DGTForEveryFrame[startIdx: endIdx].unsqueeze(0)), dim=2)[0]
    hmr_pckh05 = torch.sum(compute_pck_head(joints3DHMRForEveryFrame[startIdx: endIdx].unsqueeze(0), \
                                             joints3DGTForEveryFrame[startIdx: endIdx].unsqueeze(0)), dim=2)[0]/24

    print("aver mpjpe_result: ", torch.mean(pred_mpjpe, axis=0).item()*1000)
    print("aver pampjpe_result: ", torch.mean(pred_pa_mpjpe, axis=0).item()*1000)
    print("aver pel_mpjpe_result: ", torch.mean(pred_pelvis_mpjpe, axis=0).item()*1000)
    print("pckh@0.5: ", torch.mean(pred_pckh05, axis=0).item())
    print("------------------")
    print("aver mpjpeHMR_result: ", torch.mean(hmr_mpjpe, axis=0).item()*1000)
    print("aver pampjpeHMR_result: ", torch.mean(hmr_pa_mpjpe, axis=0).item()*1000)
    print("aver pel_mpjpeHMR_result: ", torch.mean(hmr_pelvis_mpjpe, axis=0).item()*1000)
    print("pckh@0.5: ", torch.mean(hmr_pckh05, axis=0).item())


    pdb.set_trace()







def get_args():
    def print_args(args):
        """ Prints the argparse argmuments applied
        Args:
          args = parser.parse_args()
        """
        _args = vars(args)
        max_length = max([len(k) for k, _ in _args.items()])
        for k, v in _args.items():
            print(' ' * (max_length - len(k)) + k + ': ' + str(v))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', type=str, default='/home/ziyan/02_research/EventHPE/data_event/data_event_out')
    parser.add_argument('--result_dir', type=str, default='/home/ziyan/02_research/EventHPE/exp_track')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--smpl_dir', type=str, default='/home/ziyan/02_research/EventHPE/smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
    parser.add_argument('--num_steps', type=int, default=8)
    parser.add_argument('--skip', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_optimization_epochs', type=int, default=100)
    parser.add_argument('--event_refinement_epochs', type=int, default=10)
    parser.add_argument('--joints3d_loss', type=float, default=0.25)
    parser.add_argument('--joints2d_loss', type=float, default=0.25)
    parser.add_argument('--temp_loss', type=float, default=0.1)
    parser.add_argument('--cor_loss', type=float, default=0.01)
    parser.add_argument('--stab_loss', type=float, default=1)
    parser.add_argument('--sil_loss', type=float, default=1)
    
    parser.add_argument('--startWindow', type=float, default=6) #5
    parser.add_argument('--endWindow', type=float, default=7)

    parser.add_argument('--action', type=str, default='subject01_group1_time1')
    parser.add_argument('--lr_start', '-lr', type=float, default=0.001)
    args = parser.parse_args()
    print_args(args)
    return args


def main():
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    test(args)


if __name__ == '__main__':
    main()
