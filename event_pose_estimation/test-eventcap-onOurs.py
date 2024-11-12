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
from PIL import Image
import io

def test(args):
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")

    smpl_dir = args.smpl_dir
    print('[smpl_dir] %s' % smpl_dir)

    action = 'Squat_ziyan_1017_1'
    h5 = h5py.File(f'/home/ziyan/02_research/EventHPE/{action}.h5', 'r')
    # ['events_p', 'events_t', 'events_xy', 'image_raw_event_ts']
    # f.visititems(print_dataset_size)
    image_data = np.asarray(h5['images']['binary'])
    event_trigger = np.asarray(h5['events']['event_annot_ts'])
    real_image_ts = event_trigger[::8]

    xy_u = np.asarray(h5['events']['xy_undistort'])
    # x = np.asarray(h5['events']['x'])
    # y = np.asarray(h5['events']['y'])
    t = np.asarray(h5['events']['t'])
    p = np.asarray(h5['events']['p'])

    R = np.asarray(h5['annotations']['R'])
    hmrRs = R[::8]
    T = np.asarray(h5['annotations']['T'])
    hmrTs = T[::8]
    poses = np.asarray(h5['annotations']['poses'])
    hmrPoses = poses[::8]
    shape = np.asarray(h5['annotations']['shape'])
    hmrShape = shape[::8]

    hmrParams = np.concatenate([hmrTs, hmrPoses, hmrShape], axis=1)
    gtParams = np.concatenate([T, poses, shape], axis=1)

    event_intr = np.asarray(h5['calibration']['event_intr'])
    event_extr = np.asarray(h5['calibration']['event_extr'])
    image_intr = np.asarray(h5['calibration']['image_intr'])
    image_extr = np.asarray(h5['calibration']['image_extr'])


    Rt_ei = event_extr @ np.linalg.inv(image_extr)
    R_ei = Rt_ei[:3,:3]
    H = R_ei

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

    img = Image.open(io.BytesIO(image_data[0]))
    img = np.array(img)
    image_warped = cv2.warpPerspective(np.array(img),H,dsize=(480,640))
    image_warped.shape

    numImgsInSplit = 8
    cam_intr = event_intr[1:]
    cam_intr[:2] *= 0.6 
    cam_intr = torch.from_numpy(cam_intr).float()
    Height, Width = 640, 480

    joints2DHMR_list, joints3DHMR_list, transHMR_list = [], [], []
    paramsHMR_list = []
    
    hmrParams = torch.tensor(hmrParams).to(device)
    hmr_rotmats = batch_rodrigues(hmrParams[:, 3:75].reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
    sequence_length = len(real_image_ts)
    modelSeq = SMPL(smpl_dir, sequence_length)
    # Freeze the rest of the SMPL model parameters
    for param in modelSeq.parameters():
        param.requires_grad = False  # Freeze all SMPL parameters

    modelSeq = modelSeq.to(device=device)  # move the model parameters to CPU/GPU

    _, joints3dHMR, _ = modelSeq(beta=hmrParams[:,75:85].view(-1, 10),
                            theta=None,
                            get_skin=True,
                            rotmats=hmr_rotmats.view(-1, 24, 3, 3))

    rot_h = batch_rodrigues(torch.tensor(R[:joints3dHMR.shape[0]], device=device).reshape(-1, 3))
    R_extr = torch.tensor(event_extr[:3,:3], device = device)
    T_extr = torch.tensor(event_extr[:3,-1], device = device)
    joints3dHMR_trans = torch.matmul(joints3dHMR, rot_h.transpose(1,2)) + hmrParams[:joints3dHMR.shape[0], :3].unsqueeze(1).repeat(1, joints3dHMR.shape[1], 1)
    joints3dHMR_trans = torch.matmul(joints3dHMR_trans, R_extr.T) + T_extr 

    # joints3d_trans = joints3d + learnable_pose_and_shape[:, :3].unsqueeze(1).repeat(1, joints3d.shape[1], 1)
    joints2dHMR = projection_torch(joints3dHMR_trans, cam_intr, Height, Width) * torch.tensor([Width, Height]).to(device)

    joints2DHMRForEveryFrame = joints2dHMR.to(device)
    joints3DHMRForEveryFrame = joints3dHMR_trans.to(device)
    paramsHMRForEveryFrame = hmrParams.to(device)


    gtParams = torch.tensor(gtParams).to(device)
    gt_rotmats = batch_rodrigues(gtParams[:, 3:75].reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
    sequence_length = len(event_trigger)
    modelSeq = SMPL(smpl_dir, sequence_length)
    # Freeze the rest of the SMPL model parameters
    for param in modelSeq.parameters():
        param.requires_grad = False  # Freeze all SMPL parameters

    modelSeq = modelSeq.to(device=device)  # move the model parameters to CPU/GPU

    _, joints3dGT, _ = modelSeq(beta=gtParams[:,75:85].view(-1, 10),
                            theta=None,
                            get_skin=True,
                            rotmats=gt_rotmats.view(-1, 24, 3, 3))

    rot_h = batch_rodrigues(torch.tensor(R[:joints3dGT.shape[0]], device=device).reshape(-1, 3))
    R_extr = torch.tensor(event_extr[:3,:3], device = device)
    T_extr = torch.tensor(event_extr[:3,-1], device = device)
    joints3dGT_trans = torch.matmul(joints3dGT, rot_h.transpose(1,2)) + gtParams[:joints3dGT.shape[0], :3].unsqueeze(1).repeat(1, joints3dGT.shape[1], 1)
    joints3dGT_trans = torch.matmul(joints3dGT_trans, R_extr.T) + T_extr 

    # joints3d_trans = joints3d + learnable_pose_and_shape[:, :3].unsqueeze(1).repeat(1, joints3d.shape[1], 1)
    joints2dGT = projection_torch(joints3dGT_trans, cam_intr, Height, Width) * torch.tensor([Width, Height]).to(device)

    joints2DGTForEveryFrame = joints2dGT.to(device)
    joints3DGTForEveryFrame = joints3dGT_trans.to(device)
    paramsGTForEveryFrame = gtParams.to(device)




    # Convert grayscale image to BGR format for color drawing
    image_warped
    output_image = image_warped


    # Draw dots for each feature in the data array
    tmpJoints2DHMR = joints2DHMRForEveryFrame[0].detach().cpu().numpy()
    for iJoint in range(tmpJoints2DHMR.shape[0]):
        x = int(tmpJoints2DHMR[iJoint][0])                # x coordinate
        y = int(tmpJoints2DHMR[iJoint][1])                # y coordinate
        color = [255,0,0]
        # Draw a dot on the output image
        cv2.circle(output_image, (x, y), radius=1, color=color, thickness=-1)
        cv2.imwrite('tmp1.jpg', output_image)
    


    # predParams = torch.load('learnable_parameters-V2.pt')
    predParams = torch.load('learnable_parameters-onOurs.pt')
    # predParams = torch.load('learnable_parameters-init-onOurs.pt')
    
    pred_rotmats = batch_rodrigues(predParams[:, 3:75].reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
    _, joints3dPred, _ = modelSeq(beta=predParams[:,75:85].view(-1, 10),
                            theta=None,
                            get_skin=True,
                            rotmats=pred_rotmats.view(-1, 24, 3, 3))

    rot_h = batch_rodrigues(torch.tensor(R[:joints3dPred.shape[0]], device=device).reshape(-1, 3))
    R_extr = torch.tensor(event_extr[:3,:3], device = device)
    T_extr = torch.tensor(event_extr[:3,-1], device = device)
    joints3dPred_trans = torch.matmul(joints3dPred, rot_h.transpose(1,2)) + gtParams[:joints3dPred.shape[0], :3].unsqueeze(1).repeat(1, joints3dPred.shape[1], 1)
    joints3dPred_trans = torch.matmul(joints3dPred_trans, R_extr.T) + T_extr 

    joints3DPredForEveryFrame = joints3dPred_trans.to(device)

    # joints3d_trans = joints3d + learnable_pose_and_shape[:, :3].unsqueeze(1).repeat(1, joints3d.shape[1], 1)
    joints2dPred = projection_torch(joints3dPred_trans, cam_intr, Height, Width) * torch.tensor([Width, Height]).to(device)

    joints3DHMRForEveryFrame[:len(joints3dPred_trans)].shape
    joints3dPred_trans.shape
    mpjpe_result = torch.mean(compute_mpjpe(joints3DPredForEveryFrame, joints3DHMRForEveryFrame[:len(joints3DPredForEveryFrame)]),dim=1)  # [1, T, 24]
    pa_mpjpe_result = torch.mean(compute_pa_mpjpe_eventcap(joints3DPredForEveryFrame, joints3DHMRForEveryFrame[:len(joints3DPredForEveryFrame)]), dim=1)  # [T, 24]
    pelmpjpe_result = torch.mean(compute_pelvis_mpjpe(joints3DPredForEveryFrame.unsqueeze(0), joints3DHMRForEveryFrame[:len(joints3DPredForEveryFrame)].unsqueeze(0)), dim=2).squeeze(0)
    pckh05_result = torch.sum(compute_pck_head(joints3DPredForEveryFrame.unsqueeze(0), joints3DHMRForEveryFrame[:len(joints3DPredForEveryFrame)].unsqueeze(0)), dim=2)/24
    pckh05_result = pckh05_result.squeeze(0)
    pckh05_result.shape

    startWindow = int(args.startWindow)
    endWindow = int(args.endWindow)

    import matplotlib.pyplot as plt
    mpjpe_result = mpjpe_result.cpu().detach().numpy()[startWindow*8:endWindow*8]
    pampjpe_result = pa_mpjpe_result.cpu().detach().numpy()[startWindow*8:endWindow*8]
    mpjpePel_result = pelmpjpe_result.cpu().detach().numpy()
    pckh05_result = pckh05_result.cpu().detach().numpy()

    
    print("aver mpjpe_result: ", np.mean(mpjpe_result, axis=0)*1000)
    print("aver pampjpe_result: ", np.mean(pampjpe_result, axis=0)*1000)
    print("aver pel_mpjpe_result: ", np.mean(mpjpePel_result, axis=0)*1000)
    print("pckh@0.5: ", np.mean(pckh05_result, axis=0))
    print("------------------")

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
    parser.add_argument('--smpl_dir', type=str, default='/home/ziyan/02_research/EventHPE/smpl_model/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    parser.add_argument('--num_steps', type=int, default=8)
    parser.add_argument('--skip', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_optimization_epochs', type=int, default=100)
    parser.add_argument('--event_refinement_epochs', type=int, default=10)
    parser.add_argument('--joints3d_loss', type=float, default=0.25)
    parser.add_argument('--joints2d_loss', type=float, default=0.25)
    parser.add_argument('--temp_loss', type=float, default=0.1)
    parser.add_argument('--cor_loss', type=float, default=0.01)
    parser.add_argument('--stab_loss', type=float, default=0.1)
    parser.add_argument('--sil_loss', type=float, default=20)
    
    parser.add_argument('--startWindow', type=float, default=6) #5
    parser.add_argument('--endWindow', type=float, default=7)

    parser.add_argument('--lr_start', '-lr', type=float, default=0.005)
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
