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
from event_pose_estimation.loss_funcs import compute_mpjpe, compute_pa_mpjpe, compute_pa_mpjpe_eventcap
import h5py
import random

# import event_pose_estimation.utils as util



def train(args):
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")

    smpl_dir = args.smpl_dir
    print('[smpl_dir] %s' % smpl_dir)

    # set tensorboard
    start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    writer = SummaryWriter('%s/%s/%s' % (args.result_dir, args.log_dir, start_time))
    print('[tensorboard] %s/%s/%s' % (args.result_dir, args.log_dir, start_time))

    # training
    print('------------------------------------- 3.2. Hybrid Pose Batch Optimization ------------------------------------')
    if os.path.exists('%s/%s_track%02i%02i.pkl' % (args.data_dir, 'train', args.num_steps, args.skip)):
        all_clips = pickle.load(
            open('%s/%s_track%02i%02i.pkl' % (args.data_dir, 'train', args.num_steps, args.skip), 'rb'))
    action_list = {}
    for clip in all_clips:
        action = clip[0]
        if action not in action_list:
            action_list[action] = 1
        else:
            action_list[action] +=1
    # num_interpolations = 10
    numImgsInSplit = 8
    drawImgInterval = 21
    



    # for action, sequence_length in action_list.items():
    for action, sequence_length in action_list.items():
        # set model
        modelSeq = SMPL(smpl_dir, sequence_length)
        # Freeze the rest of the SMPL model parameters
        for param in modelSeq.parameters():
            param.requires_grad = False  # Freeze all SMPL parameters

        modelSeq = modelSeq.to(device=device)  # move the model parameters to CPU/GPU

        totalSplits = sequence_length // numImgsInSplit
        with h5py.File('/home/ziyan/02_research/EventHPE/events.hdf5', 'r') as f:
            # ['events_p', 'events_t', 'events_xy', 'image_raw_event_ts']
            image_raw_event_ts = np.array(f['image_raw_event_ts'])
            events_xy = np.concatenate([np.array(f['x'])[:,np.newaxis], np.array(f['y'])[:,np.newaxis]], axis=1)
            events_ts = np.array(f['t'])
            img2events = np.searchsorted(events_ts, image_raw_event_ts)
        
        print('===== E2D and E3D initialization =====')
        # set initial SMPL parameters 
        # E2D implementation has ignored the optimization on OpenPose, just copy hmr result and interpolate to every tracking frame
        # Set initial SMPL parameters as learnable (e.g., pose and shape)
        learnable_pose_and_shape = torch.randn(totalSplits*numImgsInSplit, 85, device=device)
        interpolated_ori_trans = torch.zeros(totalSplits*numImgsInSplit,3).to(device)
        init_joints2d = torch.zeros(totalSplits*numImgsInSplit, 24, 2)
        joints2dFromHMR = torch.zeros(totalSplits, 24, 2).to(device)
        joints3dFromHMR = torch.zeros(totalSplits, 24, 3).to(device)
        
        for iSplit in range(totalSplits):
            startImg = iSplit * numImgsInSplit
            
            # SMPL trans parameters (former 3 for global translation)
            # SMPL pose parameters (middle 72 for global rotation and joint rotations)
            # SMPL shape parameters (later 10 for shape coefficients)
            if not (os.path.exists('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (args.data_dir, action, startImg))):
                    print('[hmr not exist] %s %i' % (action, startImg))
            elif not (os.path.exists('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (args.data_dir, action, startImg+numImgsInSplit-1))):
                    print('[hmr not exist] %s %i' % (action, startImg+numImgsInSplit-1))
            else:
                _, _, _params0, _tran0, _ = \
                joblib.load('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (args.data_dir, action, startImg))
                _, _, _paramsN, _tranN, _ = \
                joblib.load('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (args.data_dir, action, startImg+numImgsInSplit-1))
                _params0[:3] = _tran0
                _paramsN[:3] = _tranN
                # Create interpolation factors for each step
                alphas = np.linspace(1 / (numImgsInSplit-1), (numImgsInSplit-2) / (numImgsInSplit-1), (numImgsInSplit-2))
                # Vectorize the interpolation: add extra dimension to _params0 and _paramsN to broadcast properly
                _paramsF = (1 - alphas[:, np.newaxis]) * _params0 + alphas[:, np.newaxis] * _paramsN
                learnable_pose_and_shape[startImg:(startImg+numImgsInSplit),:] = torch.from_numpy(np.concatenate([_params0[np.newaxis,:], _paramsF, _paramsN[np.newaxis,:]], axis=0))
                _tranF = (1 - alphas[:, np.newaxis]) * _tran0 + alphas[:, np.newaxis] * _tranN
                interpolated_ori_trans[startImg:(startImg+numImgsInSplit),:] = torch.from_numpy(np.concatenate([_tran0, _tranF, _tranN], axis=0))

        learnable_pose_and_shape.requires_grad_()
        learnable_params = [learnable_pose_and_shape]
        learnable_pose_and_shape.shape
        # set optimizer
        optimizer = torch.optim.SGD(learnable_params, lr=args.lr_start, momentum=0)
        interpolated_rotmats = batch_rodrigues(learnable_pose_and_shape[:, 3:75].reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
        interpolated_rotmats.shape
        learnable_pose_and_shape[:,75:85].shape
        verts, joints3d, _ = modelSeq(beta=learnable_pose_and_shape[:,75:85].view(-1, 10),
                                theta=None,
                                get_skin=True,
                                rotmats=interpolated_rotmats.view(-1, 24, 3, 3))
        # verts would participate in Ecor 
        # joints2d would participate in Etemp
        # verts = verts + interpolated_ori_trans[startImg:(startImg+numImgsInSplit)].unsqueeze(1).repeat(1, verts.shape[1], 1)
        scale = args.img_size / 1280.
        cam_intr = np.array([1679.3, 1679.3, 641, 641]) * scale
        cam_intr = torch.from_numpy(cam_intr).float()
        H, W = args.img_size, args.img_size
        
        joints3d_trans = joints3d + interpolated_ori_trans.unsqueeze(1).repeat(1, joints3d.shape[1], 1)
        joints2d = projection_torch(joints3d_trans, cam_intr, H, W)
        # get joints 2d and 3d from HMR for E2d and E3d
        for iSplit in range(totalSplits):
            joints2dFromHMR[iSplit] = joints2d[iSplit*numImgsInSplit].clone().detach()
            joints3dFromHMR[iSplit] = joints3d_trans[iSplit*numImgsInSplit].clone().detach()
        init_joints2d = joints2d.clone().detach()
        # joints2dOnImage(joints2d, learnable_pose_and_shape.shape[0], args, action, saved_folder = "/home/ziyan/02_research/EventHPE/event_pose_estimation/init_folder")

    
        # set model
        model = SMPL(smpl_dir, numImgsInSplit)
        # Freeze the rest of the SMPL model parameters
        for param in model.parameters():
            param.requires_grad = False  # Freeze all SMPL parameters

        model = model.to(device=device)  # move the model parameters to CPU/GPU


        print('===== Img Feature Extraction =====')
        featOnSeq = []
        # Take first frame and find corners in it
        for iSplit in range(totalSplits):
            startImg = iSplit * numImgsInSplit
            if os.path.exists('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg)):
                first_gray = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            else:
                print('the first image of sequence doesnt exist')
            imgsIn1Split = first_gray[np.newaxis, :,:]

            for iImg in range(1, numImgsInSplit):
                if os.path.exists('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iImg)):
                    frame_gray = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iImg), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                else:
                    print('next image doesnt exist')
                imgsIn1Split = np.vstack([imgsIn1Split, frame_gray[np.newaxis, :,:]])
            feats = findImgFeat(imgsIn1Split)
            imgsIn1Split.shape
            for iImg in range(numImgsInSplit):      
                featOnSeq.append(feats[iImg])



        print('===== Ebatch Optimization =====')
        for iEpoch in range(args.batch_optimization_epochs):
            total_loss = []
            featImgLocs = []
            for iSplit in range(totalSplits):
                startImg = iSplit * numImgsInSplit

                # verts would participate in Ecor 
                # joints2d would participate in Etemp                
                interpolated_rotmats = batch_rodrigues(learnable_pose_and_shape[startImg:(startImg+numImgsInSplit), 3:75].reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
                interpolated_rotmats.shape
                verts, joints3d, _ = model(beta=learnable_pose_and_shape[startImg:(startImg+numImgsInSplit),75:85].view(-1, 10),
                                    theta=None,
                                    get_skin=True,
                                    rotmats=interpolated_rotmats.view(-1, 24, 3, 3))
                verts = verts + interpolated_ori_trans[startImg:(startImg+numImgsInSplit)].unsqueeze(1).repeat(1, verts.shape[1], 1)
                joints3d_trans = joints3d + interpolated_ori_trans[startImg:(startImg+numImgsInSplit)].unsqueeze(1).repeat(1, joints3d.shape[1], 1)
                joints2d = projection_torch(joints3d_trans, cam_intr, H, W)
                verts2d = projection_torch(verts, cam_intr, H, W)
                # verts2d_pix = (verts2d*H).type(torch.uint8).detach().cpu().numpy()
                verts2d_pix = (verts2d*H).detach().cpu().numpy() 

                # print('===== E2D =====')
                loss_2D = torch.tensor(0.0, requires_grad=True)
                loss_2D = loss_2D + torch.sum(torch.norm(joints2d[0] - joints2dFromHMR[iSplit], dim=1)**2)
                # print('===== E3D =====')
                loss_3D = torch.tensor(0.0, requires_grad=True)
                loss_3D = loss_3D + torch.sum(torch.norm(joints3d_trans[0] - joints3dFromHMR[iSplit], dim=1)**2)
                # print('===== Ecor =====')
                # Initialize the scalar to store the cumulative loss
                loss_cor = torch.tensor(0.0, requires_grad=True)


                for iImg in range(0, numImgsInSplit):
                    tolerance = 10
                    p0 = featOnSeq[startImg+iImg]
                    p0tensor = torch.tensor(p0, requires_grad=False).to(device)  # (n, 2)
                    min_distances, closest_vert_indices = findClosestPointTorch(verts2d[iImg]*H, p0tensor)
                    selectedDistancesSqSum = torch.sum(min_distances[min_distances < tolerance]**2)
                    loss_cor = loss_cor + selectedDistancesSqSum/H

                # print('===== Etemp =====')
                loss_temp = torch.norm((joints2d[1:] - joints2d[:-1]), dim=2).sum()


                lossFor1Seq = args.cor_loss*loss_cor + args.temp_loss*loss_temp + args.joints2d_loss*loss_2D + args.joints3d_loss*loss_3D
                
                if iSplit % drawImgInterval == 1:
                    colors = {feature_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for feature_id in range(len(p0))}
                    min_distances = min_distances.cpu().detach().numpy()
                    frame_gray = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+numImgsInSplit-1), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                    featAndVertOnImg = draw_feature_dots(frame_gray, p0[min_distances < tolerance], \
                                                        (verts2d[-1]*H).type(torch.uint8).detach().cpu().numpy(), \
                                                            closest_vert_indices[min_distances < tolerance], colors)
                    jointsForVisual = joints2d[-1].detach().cpu().numpy() * 256
                    jointsForVisual = jointsForVisual.astype(np.uint8)
                    
                    skeletonOnImg = draw_skeleton(frame_gray, jointsForVisual, draw_edges=True)

                    writer.add_images('feat and closest vertics' , featAndVertOnImg, iSplit + iEpoch*totalSplits, dataformats='HWC')
                    writer.add_images('skeleton' , skeletonOnImg, iSplit + iEpoch*totalSplits, dataformats='HWC')
                    writer.add_scalar('training_loss', lossFor1Seq.item(), iSplit + iEpoch*totalSplits)
                    writer.add_scalar('loss2D', loss_2D.item(), iSplit + iEpoch*totalSplits)
                    writer.add_scalar('loss3D', loss_3D.item(), iSplit + iEpoch*totalSplits)
                    writer.add_scalar('lossTemp', loss_temp.item(), iSplit + iEpoch*totalSplits)
                    writer.add_scalar('lossCor', loss_cor.item(), iSplit + iEpoch*totalSplits)
                    

                    

                total_loss.append(lossFor1Seq.item())
                lossFor1Seq.backward()
                learnable_params[0].grad
                optimizer.step()
                optimizer.zero_grad()
            # if (iEpoch % drawImgInterval  == 1):
            #     pdb.set_trace()

            averLossForEpoch = sum(total_loss) / len(total_loss)
            print(iEpoch, f" aver loss for Etemp & Ecor: {averLossForEpoch:.3f}")
            
            interpolated_rotmats = batch_rodrigues(learnable_pose_and_shape[:, 3:75].reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
            _, joints3dSeq, _ = modelSeq(beta=learnable_pose_and_shape[:,75:85].view(-1, 10),
                        theta=None,
                        get_skin=True,
                        rotmats=interpolated_rotmats.view(-1, 24, 3, 3))
            joints3d_transSeq = joints3dSeq + interpolated_ori_trans.unsqueeze(1).repeat(1, joints3d.shape[1], 1)
            joints2dSeq = projection_torch(joints3d_transSeq, cam_intr, H, W)

        # joints2dOnImage(joints2dSeq, learnable_pose_and_shape.shape[0], args, action, saved_folder = "/home/ziyan/02_research/EventHPE/event_pose_estimation/tmp_folder")
        # joints2dandFeatOnImage(joints2dSeq, featImgLocs, learnable_pose_and_shape.shape[0], args, action, saved_folder = "/home/ziyan/02_research/EventHPE/event_pose_estimation/tmp_folder")
            
        print('------------------------------------- 3.3. Event-Based Pose Refinement ------------------------------------')
        for iEpoch in range(args.event_refinement_epochs):
            total_loss = []

            for iSplit in range(totalSplits):
                # print('===== Esil =====')
                loss_sil = torch.tensor(0.0, requires_grad=True)
                startImg = iSplit * numImgsInSplit
                interpolated_rotmats = batch_rodrigues(learnable_pose_and_shape[startImg:(startImg+numImgsInSplit), 3:75].reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
                interpolated_rotmats.shape
                verts, joints3d, _ = model(beta=learnable_pose_and_shape[startImg:(startImg+numImgsInSplit),75:85].view(-1, 10),
                                    theta=None,
                                    get_skin=True,
                                    rotmats=interpolated_rotmats.view(-1, 24, 3, 3))
                verts = verts + interpolated_ori_trans[startImg:(startImg+numImgsInSplit)].unsqueeze(1).repeat(1, verts.shape[1], 1)
                joints3d_trans = joints3d + interpolated_ori_trans[startImg:(startImg+numImgsInSplit)].unsqueeze(1).repeat(1, joints3d.shape[1], 1)
                joints2d = projection_torch(joints3d_trans, cam_intr, H, W)
                verts2d = projection_torch(verts, cam_intr, H, W)
                # verts2d_pix = (verts2d*H).type(torch.uint8).detach().cpu().numpy()
                verts2d_pix = (verts2d*H).detach().cpu().numpy() 

                boundaryPixelsinSeq = findBoundaryPixels(verts2d_pix, 256) # len = 8
                # closest_events_u, closest_events_t = find_closest_events(boundaryPixelsinSeq[1], events_xy, events_ts, image_raw_event_ts[1], image_raw_event_ts[0], image_raw_event_ts[7], 0.5)
                for iImg in range(numImgsInSplit):
                    closest_events_u, closest_events_t = find_closest_events(boundaryPixelsinSeq[iImg], events_xy, events_ts, image_raw_event_ts[startImg+iImg], image_raw_event_ts[startImg], image_raw_event_ts[startImg+numImgsInSplit], 0.5)
                    distanceToClosestEvents = torch.norm(torch.tensor((boundaryPixelsinSeq[iImg] - closest_events_u)/H).to(torch.float16), dim = 1).to(device)
                    loss_sil = loss_sil + torch.sum((distanceToClosestEvents)**2)
                
                # print('===== Estab =====')
                loss_stab = torch.sum(torch.norm(joints2d - init_joints2d[startImg:(startImg+numImgsInSplit)], dim=2)**2)
                joints2d.shape
                init_joints2d[startImg:(startImg+numImgsInSplit)].shape
                loss_refined = args.sil_loss*loss_sil + args.stab_loss*loss_stab

                # loss_refined = loss_sil
                loss_refined.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss.append(loss_refined.item())

                if iSplit % drawImgInterval == 1:
                    if os.path.exists('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+numImgsInSplit-1)):
                        frame_gray = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+numImgsInSplit), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                    colors = {feature_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for feature_id in range(closest_events_u.shape[0])}
                    # boundaryAndEventOnImg = draw_feature_dots(frame_gray, np.transpose(boundaryPixelsinSeq[-1], (1, 0)), \
                    #                                      closest_events_u, \
                    #                                         [x for x in range(closest_events_u.shape[0])], colors)
                    boundaryAndEventOnImg = draw_feature_dots(frame_gray, boundaryPixelsinSeq[-1][:,[1,0]], closest_events_u, [x for x in range(closest_events_u.shape[0])], colors)
                    # boundaryAndEventOnImg = draw_feature_dots(frame_gray, boundaryPixelsinSeq[-1], closest_events_u, [x for x in range(closest_events_u.shape[0])], colors)
                    

                    # cv2.imwrite('tmp.jpg', boundaryAndEventOnImg)
                    writer.add_images('boundary_on_Img' , boundaryAndEventOnImg, iSplit + iEpoch*totalSplits, dataformats='HWC')
                    writer.add_scalar('refinement_loss', loss_refined.item(), iSplit + iEpoch*totalSplits)

            averLossForEpoch = sum(total_loss) / len(total_loss)
            print(iEpoch, " aver loss for Etemp & Ecor: ", averLossForEpoch)

        torch.save(learnable_pose_and_shape, 'learnable_parameters.pt')
        mpjpe_result, pampjpe_result = evaluation(args, action, learnable_pose_and_shape, model, cam_intr, device)
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
    
    parser.add_argument('--lr_start', '-lr', type=float, default=0.001)
    args = parser.parse_args()
    print_args(args)
    return args


def main():
    args = get_args()
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    train(args)


if __name__ == '__main__':
    main()
