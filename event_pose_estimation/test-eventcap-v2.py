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

    action = 'subject03_group1_time1'
    numImgsInSplit = 8
    scale = args.img_size / 1280.
    cam_intr = np.array([1679.3, 1679.3, 641, 641]) * scale
    cam_intr = torch.from_numpy(cam_intr).float()
    H, W = args.img_size, args.img_size

    joints2DHMR_list, joints3DHMR_list, transHMR_list = [], [], []
    paramsHMR_list = []
    for iImg in range(800):
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
    # output_image = image.copy()


    # joints2dSeq = projection_torch(torch.from_numpy(tmpJoints3DHMR ), cam_intr, H, W)
    # joints2dSeq = projection_torch(torch.from_numpy(tmpJoints3DHMR + np.tile(tmpTransHMR,(tmpJoints3DHMR.shape[0],1))), cam_intr, H, W)
    joints2dSeq = projection_torch(torch.from_numpy(tmpJoints3DHMR), cam_intr, H, W)
    joints2dSeq = joints2dSeq*H

    # Draw dots for each feature in the data array
    for iJoint in range(tmpJoints2DHMR.shape[0]):
        x = int(tmpJoints2DHMR[iJoint][0])                # x coordinate
        y = int(tmpJoints2DHMR[iJoint][1])                # y coordinate
        color = [255,0,0]
        # Draw a dot on the output image
        cv2.circle(output_image, (x, y), radius=1, color=color, thickness=-1)
        cv2.imwrite('tmp.jpg', output_image)
    
    
    # set model
    model = SMPL(smpl_dir, numImgsInSplit)
    # Freeze the rest of the SMPL model parameters
    for param in model.parameters():
        param.requires_grad = False  # Freeze all SMPL parameters

    model = model.to(device=device)  # move the model parameters to CPU/GPU


    learnable_pose_and_shape = torch.load('learnable_parameters-V2.pt')
    mpjpe_result, pampjpe_result, joints3DGTForEveryFrame, joints3DPredForEveryFrame = \
        evaluation(args, action, learnable_pose_and_shape, model, cam_intr, device)
    # mpjpe_result, pampjpe_result, joints3DGTForEveryFrame, joints3DPredForEveryFrame = \
    #     evaluation(args, action, paramsHMRForEveryFrame, model, cam_intr, device)

    # joints3DPredForqEveryFrame = joints3DPredForEveryFrame + transHMRForEveryFrame

    joints3DGTForEveryFrame.shape
    joints3DPredForEveryFrame.shape


    mpjpeNorm_list = []
    pampjpeNorm_list = []
    pelvis_mpjpe_list = []
    pckh05_list = []
    mpjpeHMR_list = []
    pampjpeHMR_list = []
    mpjpeHMRNorm_list = []
    pampjpeHMRNorm_list = []
    pelvis_mpjpeHMR_list = []
    pckh05HMR_list = []
    
    startWindow = 20
    endWindow = 70
    for iImg in range(startWindow, endWindow):
        joints3DGTIn1Split = joints3DGTForEveryFrame[iImg]
        joints3DPredIn1Split = joints3DPredForEveryFrame[iImg]
        joints3DPredIn1Split_Normalized = joints3DPredIn1Split - \
                    joints3DPredIn1Split[:,0,:].unsqueeze(1).repeat(1,joints3DPredIn1Split.shape[1],1)
        joints3DGTIn1Split_Normalized = joints3DGTIn1Split - \
                    joints3DGTIn1Split[:,0,:].unsqueeze(1).repeat(1,joints3DPredIn1Split.shape[1],1)
        joints3DHMRIn1Split_Normalized = joints3DHMRForEveryFrame[iImg*8:(iImg+1)*8] - joints3DHMRForEveryFrame[iImg*8:(iImg+1)*8][:,0,:].unsqueeze(1)
        mpjpe = torch.mean(compute_mpjpe(joints3DPredIn1Split_Normalized.unsqueeze(0), \
                                         joints3DGTIn1Split_Normalized.unsqueeze(0)),dim=2)  # [1, T, 24]
        pa_mpjpe = torch.mean(compute_pa_mpjpe_eventcap(joints3DPredIn1Split_Normalized,\
                                                         joints3DGTIn1Split_Normalized), dim=1)  # [T, 24]
        pelvis_mpjpe = torch.mean(compute_pelvis_mpjpe(joints3DPredIn1Split.unsqueeze(0), joints3DGTIn1Split.unsqueeze(0)), dim=2)
        
        pckh05 = torch.sum(compute_pck_head(joints3DPredIn1Split.unsqueeze(0), joints3DGTIn1Split.unsqueeze(0)), dim=2)/24

        mpjpeHMR = torch.mean(compute_mpjpe(joints3DHMRForEveryFrame[iImg*8:(iImg+1)*8].unsqueeze(0), \
                                         joints3DGTIn1Split.unsqueeze(0)),dim=2)  # [1, T, 24]
        pa_mpjpeHMR = torch.mean(compute_pa_mpjpe_eventcap(joints3DHMRForEveryFrame[iImg*8:(iImg+1)*8],\
                                                         joints3DGTIn1Split), dim=1)  # [T, 24]
        mpjpeHMR_Norm = torch.mean(compute_mpjpe(joints3DHMRIn1Split_Normalized.unsqueeze(0), \
                                         joints3DGTIn1Split_Normalized.unsqueeze(0)),dim=2)  # [1, T, 24]
        pa_mpjpeHMR_Norm = torch.mean(compute_pa_mpjpe_eventcap(joints3DHMRIn1Split_Normalized,\
                                                         joints3DGTIn1Split_Normalized), dim=1)  # [T, 24]
        pelvis_mpjpeHMR= torch.mean(compute_pelvis_mpjpe(joints3DHMRForEveryFrame[iImg*8:(iImg+1)*8].unsqueeze(0), \
                                                         joints3DGTIn1Split.unsqueeze(0)), dim=2)
        pckh05HMR = torch.sum(compute_pck_head(joints3DHMRForEveryFrame[iImg*8:(iImg+1)*8].unsqueeze(0), \
                                                joints3DGTIn1Split.unsqueeze(0)), dim=2)/24
        

        mpjpeHMR_list.append(mpjpeHMR[0])
        pampjpeHMR_list.append(pa_mpjpeHMR)
        mpjpeNorm_list.append(mpjpe[0])
        pampjpeNorm_list.append(pa_mpjpe)
        pelvis_mpjpe_list.append(pelvis_mpjpe)
        pckh05_list.append(pckh05)
        mpjpeHMRNorm_list.append(mpjpeHMR_Norm[0])
        pampjpeHMRNorm_list.append(pa_mpjpeHMR_Norm)
        pelvis_mpjpeHMR_list.append(pelvis_mpjpeHMR)
        pckh05HMR_list.append(pckh05HMR)


    mpjpeNorm_result, pampjpeNorm_result = torch.hstack(mpjpeNorm_list), torch.hstack(pampjpeNorm_list)
    mpjpeHMR_result, pampjpeHMR_result = torch.hstack(mpjpeHMR_list), torch.hstack(pampjpeHMR_list)
    mpjpeHMRNorm_result, pampjpeHMRNorm_result = torch.hstack(mpjpeHMRNorm_list), torch.hstack(pampjpeHMRNorm_list)
    mpjpePel_result, pampjpePel_result = torch.hstack(pelvis_mpjpe_list), torch.hstack(pelvis_mpjpeHMR_list)
    pckh05_result, pckh05HMR_result = torch.hstack(pckh05_list), torch.hstack(pckh05HMR_list)
    import matplotlib.pyplot as plt
    mpjpe_result = mpjpe_result.cpu().detach().numpy()[startWindow*8:endWindow*8]
    pampjpe_result = pampjpe_result.cpu().detach().numpy()[startWindow*8:endWindow*8]
    mpjpeNorm_result = mpjpeNorm_result.cpu().detach().numpy()
    pampjpeNorm_result = pampjpeNorm_result.cpu().detach().numpy()
    mpjpePel_result = mpjpePel_result[0].cpu().detach().numpy()
    pckh05_result = pckh05_result[0].cpu().detach().numpy()

    mpjpeHMR_result = mpjpeHMR_result.cpu().detach().numpy()
    pampjpeHMR_result = pampjpeHMR_result.cpu().detach().numpy()
    mpjpeHMRNorm_result = mpjpeHMRNorm_result.cpu().detach().numpy()
    pampjpeHMRNorm_result = pampjpeHMRNorm_result.cpu().detach().numpy()
    pampjpePel_result = pampjpePel_result[0].cpu().detach().numpy()
    pckh05HMR_result = pckh05HMR_result[0].cpu().detach().numpy()
    
    print("aver mpjpe_result: ", np.mean(mpjpe_result, axis=0)*1000)
    print("aver pampjpe_result: ", np.mean(pampjpe_result, axis=0)*1000)
    print("aver norm_mpjpe_result: ", np.mean(mpjpeNorm_result, axis=0)*1000)
    print("aver pel_mpjpe_result: ", np.mean(mpjpePel_result, axis=0)*1000)
    print("pckh@0.5: ", np.mean(pckh05_result, axis=0))
    print("------------------")
    print("aver mpjpeHMR_result: ", np.mean(mpjpeHMR_result, axis=0)*1000)
    print("aver pampjpeHMR_result: ", np.mean(pampjpeHMR_result, axis=0)*1000)
    print("aver norm_mpjpeHMR_result: ", np.mean(mpjpeHMRNorm_result, axis=0)*1000)
    print("aver pel_mpjpeHMR_result: ", np.mean(pampjpePel_result, axis=0)*1000)
    print("pckh@0.5: ", np.mean(pckh05HMR_result, axis=0))

    if (0):
        # Plot the tensor
        # Create a figure with 1 row and 2 columns of subplots
        fig, axes = plt.subplots(2, 2, figsize=(10, 5))
        x = np.arange(pampjpe_result.shape[0])
        x.shape
        # Plot pa_mpjpe
        axes[0,0].plot(x, pampjpe_result, color='blue')
        axes[0,0].plot(x, pampjpeHMR_result, color='red')
        axes[0,0].set_title("pampjpe")
        axes[0,0].set_xlabel("index")
        axes[0,0].set_ylabel("pampjpe")

        # Plot mpjpe
        axes[0,1].plot(x, mpjpe_result, color='blue')
        axes[0,1].plot(x, mpjpeHMR_result, color='red')
        axes[0,1].set_title("mpjpe")
        axes[0,1].set_xlabel("index")
        axes[0,1].set_ylabel("mpjpe")

        # Plot normalized pa_mpjpe
        axes[1,0].plot(x, pampjpeNorm_result, color='blue')
        axes[1,0].plot(x, pampjpeHMRNorm_result, color='red')
        axes[1,0].set_title("norm_pampjpe")
        axes[1,0].set_xlabel("index")
        axes[1,0].set_ylabel("norm_pampjpe")

        # Plot normalized mpjpe
        axes[1,1].plot(x, mpjpeNorm_result, color='blue')
        axes[1,1].plot(x, mpjpeHMRNorm_result, color='red')
        axes[1,1].set_title("norm_mpjpe")
        axes[1,1].set_xlabel("index")
        axes[1,1].set_ylabel("norm_mpjpe")

        plt.tight_layout()  # Adjusts spacing between subplots for a cleaner layout
        plt.savefig('resultforSequence.png', format='png', dpi=300)







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
