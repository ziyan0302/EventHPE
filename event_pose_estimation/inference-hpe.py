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

    action = 'subject01_group1_time1'
    saveFolder = '/home/ziyan/02_research/EventHPE/comparisons'
    sequence_length = 1345
    numImgsInSplit = 8
    scale = args.img_size / 1280.
    cam_intr = np.array([1679.3, 1679.3, 641, 641]) * scale
    cam_intr = torch.from_numpy(cam_intr).float()
    H, W = args.img_size, args.img_size
    totalSplits = sequence_length // numImgsInSplit

    joints2DHMR_list, joints3DHMR_list, transHMR_list = [], [], []
    paramsHMR_list = []

    print('===== 2D and 3D initialization =====')
    # set initial SMPL parameters 
    # E2D implementation has ignored the optimization on OpenPose, just copy hmr result and interpolate to every tracking frame
    # Set initial SMPL parameters as learnable (e.g., pose and shape)
    learnable_pose_and_shape = torch.randn(totalSplits*numImgsInSplit, 85, device=device)
    joints2dFromHMR = torch.zeros(totalSplits*numImgsInSplit, 24, 2).to(device)
    joints3dFromHMR = torch.zeros(totalSplits*numImgsInSplit, 24, 3).to(device)
    transFromHMR = torch.zeros(totalSplits*numImgsInSplit,3).to(device)

    
    for iSplit in range(totalSplits):
        startImg = iSplit * numImgsInSplit
        
        # SMPL trans parameters (former 3 for global translation)
        # SMPL pose parameters (middle 72 for global rotation and joint rotations)
        # SMPL shape parameters (later 10 for shape coefficients)
        if not (os.path.exists('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (args.data_dir, action, startImg))):
                print('[hmr not exist] %s %i' % (action, startImg))
        elif not (os.path.exists('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (args.data_dir, action, startImg+numImgsInSplit))):
                print('[hmr not exist] %s %i' % (action, startImg+numImgsInSplit))
        else:
            _, _, _params0, _tran0, _ = \
            joblib.load('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (args.data_dir, action, startImg))
            _, _, _paramsN, _tranN, _ = \
            joblib.load('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (args.data_dir, action, startImg+numImgsInSplit))
            _params0[:3] = _tran0
            _paramsN[:3] = _tranN
            # Create interpolation factors for each step
            alphas = np.linspace(1 / numImgsInSplit, (numImgsInSplit-1) / numImgsInSplit, (numImgsInSplit-1))
            # Vectorize the interpolation: add extra dimension to _params0 and _paramsN to broadcast properly
            _paramsF = (1 - alphas[:, np.newaxis]) * _params0 + alphas[:, np.newaxis] * _paramsN
            # _paramsF = (_paramsF+_params0)/2
            learnable_pose_and_shape[startImg:(startImg+numImgsInSplit),:] = torch.from_numpy(np.concatenate([_params0[np.newaxis,:], _paramsF], axis=0))
            pdb.set_trace()


    # learnable_pose_and_shape.shape

    interpolated_rotmats = batch_rodrigues(learnable_pose_and_shape[:, 3:75].reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
    interpolated_rotmats.shape
    learnable_pose_and_shape[:,75:85].shape
    # set model
    modelSeq = SMPL(smpl_dir, sequence_length)
    # Freeze the rest of the SMPL model parameters
    for param in modelSeq.parameters():
        param.requires_grad = False  # Freeze all SMPL parameters

    modelSeq = modelSeq.to(device=device)  # move the model parameters to CPU/GPU

    init_verts, init_joints3d, _ = modelSeq(beta=learnable_pose_and_shape[:,75:85].view(-1, 10),
                            theta=None,
                            get_skin=True,
                            rotmats=interpolated_rotmats.view(-1, 24, 3, 3))
    joints3d_trans = init_joints3d + learnable_pose_and_shape[:, :3].unsqueeze(1).repeat(1, init_joints3d.shape[1], 1)
    init_joints2d = projection_torch(joints3d_trans, cam_intr, H, W).clone().detach()
    
    print('===== HMR Parameters =====')
    for iSplit in range(totalSplits):
        startImg = iSplit * numImgsInSplit
        for iImg in range(numImgsInSplit):
            joint2DHMR, joint3DHMR, _, transHMR, _ = \
                joblib.load('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (args.data_dir, action, startImg + iImg))
            joints2dFromHMR[startImg + iImg] = torch.from_numpy(joint2DHMR/H)
            joints3dFromHMR[startImg + iImg] = torch.from_numpy(joint3DHMR + np.tile(transHMR,(joint3DHMR.shape[0],1)))
            transFromHMR[startImg + iImg] = torch.from_numpy(transHMR)




    print('===== Load Pretrained Parameters =====')
    learnable_pose_and_shape = torch.load('learnable_parameters-hpegood.pt')
    rotmats = batch_rodrigues(learnable_pose_and_shape[:, 3:75].reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
    predVerts, predJoints3d, _ = modelSeq(beta=learnable_pose_and_shape[:,75:85].view(-1, 10),
                            theta=None,
                            get_skin=True,
                            rotmats=rotmats.view(-1, 24, 3, 3))
    predJoints3d_trans = predJoints3d + learnable_pose_and_shape[:, :3].unsqueeze(1).repeat(1, predJoints3d.shape[1], 1)
    pred_joints2d = projection_torch(predJoints3d_trans, cam_intr, H, W).clone().detach()

    print('===== Draw Comparison Images =====')
    for iSplit in range(int(args.startWindow),int(args.endWindow)):
        startImg = iSplit * numImgsInSplit
        comparisonImg = np.zeros((H, W*3, 3))
        for iImg in range(numImgsInSplit):
            # Convert grayscale image to BGR format for color drawing
            grey_img = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iImg), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            output_image = cv2.cvtColor(grey_img, cv2.COLOR_GRAY2BGR)
            # output_image = image.copy()

            initImg = grey_img.copy()
            init_joints2dPix = init_joints2d[startImg+iImg].detach().cpu().numpy() * np.array([W, H])
            init_joints2dPix = init_joints2dPix.astype(np.uint8)
            initImg = draw_skeleton(initImg, init_joints2dPix)
            initImg = putFontonImg('init', initImg)
            comparisonImg[:, :W, :] = initImg


            predImg = output_image.copy()
            # predVertsfor1frame = projection_torch(predVerts[startImg+iImg], cam_intr, H, W).detach().cpu()
            # predVertsfor1frame = (predVertsfor1frame * torch.tensor([W, H])).numpy().astype(np.uint8)
            # predVertsfor1frame = np.clip(predVertsfor1frame, 0, [W-1,H-1])
            # predImg[predVertsfor1frame[:,1], predVertsfor1frame[:,0]] = [255,0,0]
            pred_joints2dPix = pred_joints2d[startImg+iImg].detach().cpu().numpy() * np.array([W, H])
            pred_joints2dPix = pred_joints2dPix.astype(np.uint8)
            predImg = draw_skeleton(predImg, pred_joints2dPix)
            predImg = putFontonImg('pred', predImg)
            comparisonImg[:, W:W*2, :] = predImg


            hmrImg = grey_img.copy()
            hmr_joints2dPix = joints2dFromHMR[startImg+iImg].detach().cpu().numpy() * np.array([W, H])
            hmr_joints2dPix = hmr_joints2dPix.astype(np.uint8)
            hmrImg = draw_skeleton(hmrImg, hmr_joints2dPix)
            hmrImg = putFontonImg('hmr', hmrImg)
            comparisonImg[:, W*2:W*3, :] = hmrImg
            
            cv2.imwrite(os.path.join(saveFolder, f'{(startImg+iImg):04d}.jpg'), comparisonImg)


def putFontonImg(text, image):
    # Define font, scale, and color
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1  # Size of the text
    color = (255, 255, 255)  # White color (in BGR format)
    thickness = 2  # Thickness of the text
    # Specify position (top-left corner of the text)
    position = (50, 200)  # (x, y)

    # Write the text on the image
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    return image

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
