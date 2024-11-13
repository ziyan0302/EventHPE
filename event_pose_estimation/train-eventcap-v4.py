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
    joints2dandFeatOnImage, draw_feature_dots, draw_skeleton, findImgFeat, \
    evaluation, findBoundaryPixels_torch, find_closest_events_torch, \
    vertices_to_silhouette, draw_feature_dots_lines, find_closest_events_torch_v2
from event_pose_estimation.geometry import projection_torch, rot6d_to_rotmat, delta_rotmat_to_rotmat
from event_pose_estimation.loss_funcs import compute_mpjpe, compute_pa_mpjpe, compute_pa_mpjpe_eventcap
import h5py
import random
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


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
    drawImgInterval = 100
    



    # for action, sequence_length in action_list.items():
    for action, sequence_length in action_list.items():
        action = 'subject01_group1_time1'
        sequence_length = 1345
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

        learnable_pose_and_shape.requires_grad_()
        learnable_params = [learnable_pose_and_shape]
        # learnable_pose_and_shape.shape

        interpolated_rotmats = batch_rodrigues(learnable_pose_and_shape[:, 3:75].reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
        interpolated_rotmats.shape
        learnable_pose_and_shape[:,75:85].shape
        verts, joints3d, _ = modelSeq(beta=learnable_pose_and_shape[:,75:85].view(-1, 10),
                                theta=None,
                                get_skin=True,
                                rotmats=interpolated_rotmats.view(-1, 24, 3, 3))

        scale = args.img_size / 1280.
        cam_intr = np.array([1679.3, 1679.3, 641, 641]) * scale
        cam_intr = torch.from_numpy(cam_intr).float()
        H, W = args.img_size, args.img_size
        
        joints3d_trans = joints3d + learnable_pose_and_shape[:, :3].unsqueeze(1).repeat(1, joints3d.shape[1], 1)
        joints2d = projection_torch(joints3d_trans, cam_intr, H, W)
        init_joints2d = joints2d.clone().detach()

        # get joints 2d and 3d from HMR for E2d and E3d
        # get transHMR from every frame
        transHMREveryFrame = torch.zeros(totalSplits*numImgsInSplit,3).to(device)
        for iSplit in range(totalSplits):
            joint2DHMR, joint3DHMR, _, transHMR, _ = \
                joblib.load('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (args.data_dir, action, iSplit*numImgsInSplit))
            joints2dFromHMR[iSplit] = torch.from_numpy(joint2DHMR/H)
            joints3dFromHMR[iSplit] = torch.from_numpy(joint3DHMR + np.tile(transHMR,(joint3DHMR.shape[0],1)))
            for iImg in range(numImgsInSplit):
                _, _, _, transHMRInOneFrame, _ = \
                    joblib.load('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (args.data_dir, action, iSplit*numImgsInSplit + iImg))
                transHMREveryFrame[iSplit*numImgsInSplit + iImg] = torch.from_numpy(transHMRInOneFrame)
        # joints2dOnImage(joints2d, learnable_pose_and_shape.shape[0], args, action, saved_folder = "/home/ziyan/02_research/EventHPE/event_pose_estimation/init_folder")

    
        # set model
        model = SMPL(smpl_dir, numImgsInSplit+1)
        # Freeze the rest of the SMPL model parameters
        for param in model.parameters():
            param.requires_grad = False  # Freeze all SMPL parameters

        model = model.to(device=device)  # move the model parameters to CPU/GPU


        print('===== Img Feature Extraction =====')
        featsEveryFrame = []
        featOnEveryBackward = []
        featOnEveryForward = []

        for iSplit in range(totalSplits):
            
            # Forward Feature Extraction and backward feature extraction 
            # Forward: Take first frame and find corners in it; Backward: Take the last frame
            startImg = iSplit * numImgsInSplit
            
            if os.path.exists('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg)):
                first_gray = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
            else:
                print('the first image of sequence doesnt exist')
            imgsIn1Split = first_gray[np.newaxis, :,:]

            for iImg in range(1, numImgsInSplit+1):
                if os.path.exists('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iImg)):
                    frame_gray = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iImg), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                else:
                    print('next image doesnt exist')
                imgsIn1Split = np.vstack([imgsIn1Split, frame_gray[np.newaxis, :,:]])
            featsForward = findImgFeat(imgsIn1Split) # 0->8
            featsBackward = findImgFeat(imgsIn1Split[::-1]) # 8->0
            
            # stitching the bidirectional features
            featsBackwardReversed = featsBackward[::-1] 

            # add iImg == 0
            distances = distance_matrix(featsForward[0], featsBackwardReversed[0])
            row_indices, col_indices = linear_sum_assignment(distances)
            matched_forward = featsForward[0][row_indices]
            matched_backward = featsBackwardReversed[0][col_indices]
            matched_mean = (matched_forward + matched_backward)/2
            stitch_mask = np.linalg.norm(matched_forward - matched_backward, axis=1) < 4
            featsEveryFrame.append(matched_mean[stitch_mask])
            featOnEveryBackward.append(featsBackwardReversed[1][col_indices][stitch_mask])
            
            for iImg in range(1,numImgsInSplit):    
                distances = distance_matrix(featsForward[iImg], featsBackwardReversed[iImg])
                # Apply the Hungarian algorithm to find the optimal one-to-one correspondence
                row_indices, col_indices = linear_sum_assignment(distances)

                matched_forward = featsForward[iImg][row_indices]
                matched_backward = featsBackwardReversed[iImg][col_indices]
                matched_mean = (matched_forward + matched_backward)/2
                stitch_mask = np.linalg.norm(matched_forward - matched_backward, axis=1) < 4
                if (0):
                    pdb.set_trace()
                    tmpP = matched_mean.astype(np.uint8)
                    tmpPF = matched_forward.astype(np.uint8)
                    tmpPB = matched_backward.astype(np.uint8)
                    
                    frame_gray = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iImg), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                    colors = {feature_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for feature_id in range(len(tmpP))}
                    img = draw_feature_dots_lines(imgsIn1Split[iImg-1], tmpP, tmpP, np.arange(len(tmpP)), colors)
                    img = draw_feature_dots_lines(frame_gray, tmpP, tmpP, np.arange(len(tmpP)), colors)
                    img = draw_feature_dots(imgsIn1Split[iImg], featsForward[iImg-1], tmpPF, np.arange(len(tmpP)), colors)
                    img = draw_feature_dots(imgsIn1Split[iImg], featsForward[iImg+1], tmpPF, np.arange(len(tmpP)), colors)
                    img = draw_feature_dots_lines(frame_gray, tmpPF, tmpPF, np.arange(len(tmpP)), colors)
                    img = draw_feature_dots_lines(frame_gray, tmpPB, tmpPB, np.arange(len(tmpP)), colors)
                    cv2.imwrite("tmp.jpg", img)
                featsEveryFrame.append(matched_mean[stitch_mask])
                featOnEveryForward.append(featsForward[iImg-1][row_indices][stitch_mask])
                featOnEveryBackward.append(featsBackwardReversed[iImg+1][col_indices][stitch_mask])

            # add iImg == numImgsInSplit-1
            distances = distance_matrix(featsForward[numImgsInSplit], featsBackwardReversed[numImgsInSplit])
            row_indices, col_indices = linear_sum_assignment(distances)
            matched_forward = featsForward[numImgsInSplit][row_indices]
            matched_backward = featsBackwardReversed[numImgsInSplit][col_indices]
            matched_mean = (matched_forward + matched_backward)/2
            stitch_mask = np.linalg.norm(matched_forward - matched_backward, axis=1) < 4
            featsEveryFrame.append(matched_mean[stitch_mask])
            featOnEveryForward.append(featsForward[numImgsInSplit-1][row_indices][stitch_mask])
            
        # pdb.set_trace()
        len(featsEveryFrame)
        len(featOnEveryForward)
        len(featOnEveryBackward)
                
        print('===== Ebatch Optimization =====')
        # for iSplit in range(totalSplits):
        # set optimizer
        print(time.time())
        optimizer = torch.optim.SGD(learnable_params, lr=args.lr_start, momentum=0)
        # optimizer = torch.optim.Adam(learnable_params, lr=args.lr_start)
        for iSplit in range(int(args.startWindow),int(args.endWindow)):
            transComparison_list = []
            for iEpoch in range(args.batch_optimization_epochs):
                startImg = iSplit * numImgsInSplit

                # verts would participate in Ecor 
                # joints2d would participate in Etemp                
                rotmats = batch_rodrigues(learnable_pose_and_shape[startImg:(startImg+numImgsInSplit+1), 3:75].reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
                verts, joints3d, _ = model(beta=learnable_pose_and_shape[startImg:(startImg+numImgsInSplit+1),75:85].view(-1, 10),
                                    theta=None,
                                    get_skin=True,
                                    rotmats=rotmats.view(-1, 24, 3, 3))
                verts = verts + learnable_pose_and_shape[startImg:(startImg+numImgsInSplit+1), :3].unsqueeze(1).repeat(1, verts.shape[1], 1)
                joints3d_trans = joints3d + learnable_pose_and_shape[startImg:(startImg+numImgsInSplit+1), :3].unsqueeze(1).repeat(1, joints3d.shape[1], 1)
                joints2d = projection_torch(joints3d_trans, cam_intr, H, W)
                verts2d = projection_torch(verts, cam_intr, H, W)

                # transComparison_list.append(transHMREveryFrame[startImg:(startImg+numImgsInSplit+1)] - learnable_pose_and_shape[startImg:(startImg+numImgsInSplit+1), :3])
                # if iEpoch % drawImgInterval == 1:
                #     print(f'---{iEpoch} trans diff to HMR ---')
                #     print(torch.mean(torch.stack(transComparison_list), dim=0))
                #     print('-------------')
                #     transComparison_list = []

                # print('===== E2D =====')
                loss_2D = torch.tensor(0.0, requires_grad=True)
                loss_2D = loss_2D + torch.sum(torch.norm(joints2d[0] - joints2dFromHMR[iSplit], dim=1)**2)
                loss_2D = loss_2D + torch.sum(torch.norm(joints2d[-1] - joints2dFromHMR[iSplit+1], dim=1)**2)
                # print('===== E3D =====')
                loss_3D = torch.tensor(0.0, requires_grad=True)
                loss_3D = loss_3D + torch.sum(torch.norm(joints3d_trans[0] - joints3dFromHMR[iSplit], dim=1)**2)
                loss_3D = loss_3D + torch.sum(torch.norm(joints3d_trans[-1] - joints3dFromHMR[iSplit+1], dim=1)**2)
                # print('===== Ecor =====')
                # Initialize the scalar to store the cumulative loss
                loss_cor = torch.tensor(0.0, requires_grad=True)

                tolerance = 6
                dist_tolerance = 30
                feat_tolerance = 50
                startFeatFrame = iSplit*(numImgsInSplit-1)

                imgCorFor1Frame = np.zeros((frame_gray.shape[0]*(numImgsInSplit+1), frame_gray.shape[1]*3, 3))
                pStart = iSplit * (numImgsInSplit+1)
                for iP in range(0, numImgsInSplit+1):
                    if iP == 0:
                        pCurr = torch.tensor(featsEveryFrame[pStart + iP], requires_grad=False).to(device)  # (n, 2)
                        rough_constraint_for_feat = init_joints2d[startImg + iP] * torch.tensor([W, H]).to(device)
                        distances = torch.norm(pCurr.unsqueeze(1) - rough_constraint_for_feat.unsqueeze(0), dim=2)  # Resulting shape (N, M)                
                        feature_around_body_mask = torch.min(distances,dim=1)[0] < feat_tolerance
                        min_distances, closest_vert_indices = findClosestPointTorch(verts2d[iP]*H, pCurr[feature_around_body_mask])
                        valid_mask = min_distances < tolerance

                        # the same feat in the next frame and the same closest vertics in the next frame
                        pNext = torch.tensor(featOnEveryBackward[startImg + iP], requires_grad=False).to(device)[feature_around_body_mask][valid_mask]  # (n, 2)
                        vNext = verts2d[iP+1,closest_vert_indices][valid_mask]*H
                        nextDistances = pNext - vNext
                        valid_dist_mask = torch.norm(nextDistances, dim=1) < dist_tolerance
                        totalDistances = nextDistances[valid_dist_mask]
                        selectedDistancesSqSum = torch.sum(torch.norm(totalDistances/H, dim=1)**2)

                    elif iP == numImgsInSplit:
                        pCurr = torch.tensor(featsEveryFrame[pStart + iP], requires_grad=False).to(device)  # (n, 2)
                        rough_constraint_for_feat = init_joints2d[startImg + iP] * torch.tensor([W, H]).to(device)
                        distances = torch.norm(pCurr.unsqueeze(1) - rough_constraint_for_feat.unsqueeze(0), dim=2)  # Resulting shape (N, M)                
                        feature_around_body_mask = torch.min(distances,dim=1)[0] < feat_tolerance
                        min_distances, closest_vert_indices = findClosestPointTorch(verts2d[iP]*H, pCurr[feature_around_body_mask])
                        valid_mask = min_distances < tolerance

                        # the same feat in the last frame and the same closest vertics in the last frame
                        pLast = torch.tensor(featOnEveryForward[startImg + iP-1], requires_grad=False).to(device)[feature_around_body_mask][valid_mask]  # (n, 2)
                        vLast = verts2d[iP-1,closest_vert_indices][valid_mask]*H
                        lastDistances = pLast - vLast
                        valid_dist_mask = torch.norm(lastDistances, dim=1) < dist_tolerance
                        totalDistances = lastDistances[valid_dist_mask]
                        selectedDistancesSqSum = torch.sum(torch.norm(totalDistances/H, dim=1)**2)

                    else:
                        pCurr = torch.tensor(featsEveryFrame[pStart + iP], requires_grad=False).to(device)  # (n, 2)
                        rough_constraint_for_feat = init_joints2d[startImg + iP] * torch.tensor([W, H]).to(device)
                        distances = torch.norm(pCurr.unsqueeze(1) - rough_constraint_for_feat.unsqueeze(0), dim=2)  # Resulting shape (N, M)                
                        feature_around_body_mask = torch.min(distances,dim=1)[0] < feat_tolerance
                        min_distances, closest_vert_indices = findClosestPointTorch(verts2d[iP]*H, pCurr[feature_around_body_mask])
                        
                        # tau(i,h) filter out the non close feature
                        valid_mask = min_distances < tolerance

                        # the same feat in the next frame and the same closest vertics in the next frame
                        featOnEveryBackward[pStart + iP].shape
                        pCurr.shape
                        pNext = torch.tensor(featOnEveryBackward[startImg + iP], requires_grad=False).to(device)[feature_around_body_mask][valid_mask]  # (n, 2)
                        vNext = verts2d[iP+1,closest_vert_indices][valid_mask]*H
                        nextDistances = pNext - vNext
                        valid_dist_mask = torch.norm(nextDistances, dim=1) < dist_tolerance
                        nextDistances = torch.sum(torch.norm(nextDistances[valid_dist_mask]/H, dim=1)**2)

                        # the same feat in the last frame and the same closest vertics in the last frame
                        pLast = torch.tensor(featOnEveryForward[startImg + iP-1], requires_grad=False).to(device)[feature_around_body_mask][valid_mask]  # (n, 2)
                        vLast = verts2d[iP-1,closest_vert_indices][valid_mask]*H
                        lastDistances = pLast - vLast
                        valid_dist_mask = torch.norm(lastDistances, dim=1) < dist_tolerance
                        lastDistances = torch.sum(torch.norm(lastDistances[valid_dist_mask]/H, dim=1)**2)

                        selectedDistancesSqSum = nextDistances + lastDistances 
                    if args.drawImage and (iEpoch % drawImgInterval == 0):
                        # tmpP = pCurr[valid_mask].clone().detach().cpu().numpy().astype(np.uint8)
                        # tmpV = (verts2d[iP+1]*H)[closest_vert_indices][valid_mask].detach().cpu().numpy().astype(np.uint8)
                        tmpP = pCurr[feature_around_body_mask][valid_mask].clone().detach().cpu().numpy()
                        tmpV = (verts2d[iP]*H)[closest_vert_indices][valid_mask].detach().cpu().numpy()
                        frame_gray = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iP), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                        # colors = {feature_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for feature_id in range(len(tmpP))}
                        img = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
                        img = draw_feature_dots_lines(img, tmpP, tmpV, np.arange(len(tmpP)), H, W)
                        # feat: [255,255,0]; vert: [0,255,255]
                        tmpVAll = np.clip((verts2d[iP]*H).detach().cpu().numpy(), 0, H).astype(np.uint8)
                        img[tmpVAll[:,1], tmpVAll[:,0]] = [0,0,255]

                        imgCorFor1Frame[ iP*H: (iP+1)*H, W:W*2, :] = img
                        # cv2.imwrite("tmp.jpg", img)
                        if iP < numImgsInSplit:
                            # tmpNextP = pNext.clone().detach().cpu().numpy().astype(np.uint8)
                            # tmpNextV = (verts2d[iP+2]*H)[closest_vert_indices][valid_mask].detach().cpu().numpy().astype(np.uint8)
                            tmpNextP = pNext.clone().detach().cpu().numpy()
                            tmpNextV = (verts2d[iP+1]*H)[closest_vert_indices][valid_mask].detach().cpu().numpy()
                            frame_grayNext = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iP+1), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                            # colors = {feature_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for feature_id in range(len(tmpNextP))}
                            img = cv2.cvtColor(frame_grayNext, cv2.COLOR_GRAY2BGR)
                            img = draw_feature_dots_lines(img, tmpNextP, tmpNextV, np.arange(len(tmpNextP)), H, W)
                            tmpVAll = np.clip((verts2d[iP+1]*H).detach().cpu().numpy(), 0, H).astype(np.uint8)
                            img[tmpVAll[:,1], tmpVAll[:,0]] = [0,0,255]
                            imgCorFor1Frame[ iP*H: (iP+1)*H, W*2:W*3, :] = img
                            # cv2.imwrite("tmp.jpg", img)
                        if iP > 0:
                            tmpLastP = pLast.clone().detach().cpu().numpy().astype(np.uint8)
                            tmpLastV = (verts2d[iP-1]*H)[closest_vert_indices][valid_mask].detach().cpu().numpy()
                            frame_grayLast = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iP-1), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                            # colors = {feature_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for feature_id in range(len(tmpLastP))}
                            img = cv2.cvtColor(frame_grayLast, cv2.COLOR_GRAY2BGR)
                            img = draw_feature_dots_lines(img, tmpLastP, tmpLastV, np.arange(len(tmpLastP)), H, W)
                            tmpVAll = np.clip((verts2d[iP-1]*H).detach().cpu().numpy(), 0, H).astype(np.uint8)
                            img[tmpVAll[:,1], tmpVAll[:,0]] = [0,0,255]
                            imgCorFor1Frame[ iP*H: (iP+1)*H, :W, :] = img
                            
                            # cv2.imwrite("tmp.jpg", imgCorFor1Frame)
                

                    loss_cor = loss_cor + selectedDistancesSqSum
                
                if (0):
                    iP = 0
                    tmpCorFor1Frame = np.zeros((frame_gray.shape[0], frame_gray.shape[1]*3, 3))
                    tmpP = pCurr[feature_around_body_mask][valid_mask].clone().detach().cpu().numpy()
                    tmpV = (verts2d[iP+1]*H)[closest_vert_indices][valid_mask].detach().cpu().numpy()
                    print(iEpoch)
                    frame_gray = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iP+1), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                    # colors = {feature_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for feature_id in range(len(tmpP))}
                    img = draw_feature_dots_lines(frame_gray, tmpP, tmpV, np.arange(len(tmpP)), H, W)
                    # feat: [255,255,0]; vert: [0,255,255]
                    tmpVAll = np.clip((verts2d[iP+1]*H).detach().cpu().numpy(), 0, H).astype(np.uint8)
                    img[tmpVAll[:,1], tmpVAll[:,0]] = [0,0,255]
                    tmpCorFor1Frame[:, W:W*2, :] = img


                    tmpNextP = pNext.clone().detach().cpu().numpy()
                    tmpNextV = (verts2d[iP+2]*H)[closest_vert_indices][valid_mask].detach().cpu().numpy()
                    frame_grayNext = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iP+2), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                    # colors = {feature_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for feature_id in range(len(tmpNextP))}
                    img = draw_feature_dots_lines(frame_grayNext, tmpNextP, tmpNextV, np.arange(len(tmpNextP)), H, W)
                    tmpVAll = np.clip((verts2d[iP+2]*H).detach().cpu().numpy(), 0, H).astype(np.uint8)
                    img[tmpVAll[:,1], tmpVAll[:,0]] = [0,0,255]
                    tmpCorFor1Frame[:, W*2:W*3, :] = img

                    
                    tmpLastP = pLast.clone().detach().cpu().numpy().astype(np.uint8)
                    tmpLastV = (verts2d[iP]*H)[closest_vert_indices][valid_mask].detach().cpu().numpy().astype(np.uint8)
                    tmpLastP = pLast.clone().detach().cpu().numpy()
                    tmpLastV = (verts2d[iP]*H)[closest_vert_indices][valid_mask].detach().cpu().numpy()
                    frame_grayLast = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iP), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                    # colors = {feature_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for feature_id in range(len(tmpLastP))}
                    img = draw_feature_dots_lines(frame_grayLast, tmpLastP, tmpLastV, np.arange(len(tmpLastP)), H, W)
                    tmpVAll = np.clip((verts2d[iP]*H).detach().cpu().numpy(), 0, H).astype(np.uint8)
                    img[tmpVAll[:,1], tmpVAll[:,0]] = [0,0,255]
                    tmpCorFor1Frame[:, :W, :] = img
                    cv2.imwrite('tmp.jpg', tmpCorFor1Frame)
                    time.sleep(0.5)
                # print('===== Etemp =====')
                loss_temp = torch.tensor(0.0, requires_grad=True)
                if (1):
                    toleranceTemp = 10
                    imgTempToShow = np.zeros((H, W*(numImgsInSplit+1), 3))
                    for iImg in range(0, numImgsInSplit):
                        if iImg == 0:
                            p0 = torch.tensor(featOnEveryForward[startFeatFrame], requires_grad=False).to(device)  # (n, 2)
                        elif iImg == numImgsInSplit:
                            p0 = torch.tensor(featOnEveryBackward[startFeatFrame+numImgsInSplit-2], requires_grad=False).to(device)  # (n, 2)
                        else:
                            p0 = torch.tensor(featsEveryFrame[startFeatFrame+iImg-1], requires_grad=False).to(device)  # (n, 2)

                        startEventIdx = max(img2events[startImg+iImg]-1000, 0)
                        endEventIdx = min(img2events[startImg+iImg]+1000, len(events_xy))
                        events_xys =  events_xy[startEventIdx:endEventIdx].astype(np.int16)
                        events_xys = torch.from_numpy(events_xys).to(device)

                        joints2dPix = joints2d[iImg] * torch.tensor([W, H], device=device)
                        # calculate distance to each feature points
                        distances = torch.norm(joints2dPix.unsqueeze(1) - events_xys.unsqueeze(0), dim=2)  # Resulting shape (N, M)
                        selectedJoints = torch.any(torch.topk(distances, dim=1, k = 15, largest=False)[0] > toleranceTemp, dim=1)
                        loss_temp = loss_temp + torch.norm((joints3d[iImg][selectedJoints] - joints3d[iImg+1][selectedJoints]), dim=1).sum()

                        if args.drawImage and (iEpoch % drawImgInterval == 0):
                            img = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iImg), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                            img = np.array(img)
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            
                            tmpJ = joints2dPix[selectedJoints].detach().cpu().numpy().astype(np.uint16)
                            tmpJAll = joints2dPix.detach().cpu().numpy().astype(np.uint16)
                            tmpE = events_xys.detach().cpu().numpy()
                            img[tmpE[:,1], tmpE[:,0]] = [0,255,0]
                            
                            for iJ in range(len(tmpJAll)):
                                cv2.circle(img, (tmpJAll[iJ,0], tmpJAll[iJ,1]), 3, [255,0,0], -1)
                            for iJ in range(len(tmpJ)):
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                font_scale = 0.3  # Size of the text
                                selectedID = np.array([range(len(joints3d[iImg]))]).T[selectedJoints.detach().cpu().numpy() == True]
                                selectedID[0]
                                cv2.putText(img, str(selectedID[iJ].item()), (tmpJ[iJ,0]+3, tmpJ[iJ,1]+3), font, font_scale, [0,0,255], 1)
                                cv2.circle(img, (tmpJ[iJ,0], tmpJ[iJ,1]), 3, [0,0,255], -1)
                            imgTempToShow[:, iImg*W: (iImg+1)*W, :] = img
                            # cv2.imwrite("tmp.jpg", img)
                            # pdb.set_trace()


                        if (0):
                            
                            for tmpImg in range(numImgsInSplit):
                                startEventIdx = max(img2events[startImg+tmpImg]-1000, 0)
                                endEventIdx = min(img2events[startImg+tmpImg]+1000, len(events_xy))
                                events_xys =  np.clip(events_xy[startEventIdx:endEventIdx], 0, H-1).astype(np.int16)
                                events_xys = torch.from_numpy(events_xys).to(device)

                                img = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+tmpImg), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                                img = np.array(img)
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                                tmpJ = joints2dPix.detach().cpu().numpy().astype(np.uint16)
                                selectedJ = tmpJ[selectedJoints.detach().cpu().numpy()]
                                tmpE = np.clip(events_xys.detach().cpu().numpy().astype(np.uint16), 0, [W-2, H-2])
                                img[tmpE[:,1], tmpE[:,0]] = [255,0,255]
                                img[tmpE[:,1]+1, tmpE[:,0]] = [255,0,255]
                                img[tmpE[:,1]-1, tmpE[:,0]] = [255,0,255]
                                img[tmpE[:,1], tmpE[:,0]+1] = [255,0,255]
                                img[tmpE[:,1], tmpE[:,0]-1] = [255,0,255]

                                startEventIdx = max(img2events[startImg+tmpImg+1]-1000, 0)
                                endEventIdx = min(img2events[startImg+tmpImg+1]+1000, len(events_xy))
                                events_ts[endEventIdx]
                                events_xys =  np.clip(events_xy[startEventIdx:endEventIdx], 0, H-1).astype(np.int16)
                                events_xys = torch.from_numpy(events_xys).to(device)
                                tmpE = np.clip(events_xys.detach().cpu().numpy().astype(np.uint16), 0, [W-2, H-2])
                                img[tmpE[:,1], tmpE[:,0]] = [255,0,0]
                                img[tmpE[:,1]+1, tmpE[:,0]] = [255,0,0]
                                img[tmpE[:,1]-1, tmpE[:,0]] = [255,0,0]
                                img[tmpE[:,1], tmpE[:,0]+1] = [255,0,0]
                                img[tmpE[:,1], tmpE[:,0]-1] = [255,0,0]

                                img[tmpJ[:,1], tmpJ[:,0]] = [0,0,255]
                                img[tmpJ[:,1]+1, tmpJ[:,0]] = [0,0,255]
                                img[tmpJ[:,1]-1, tmpJ[:,0]] = [0,0,255]
                                img[tmpJ[:,1], tmpJ[:,0]+1] = [0,0,255]
                                img[tmpJ[:,1], tmpJ[:,0]-1] = [0,0,255]
                                img[selectedJ[:,1], selectedJ[:,0]] = [0,255,255]
                                img[selectedJ[:,1]+1, selectedJ[:,0]] = [0,255,255]
                                img[selectedJ[:,1]-1, selectedJ[:,0]] = [0,255,255]
                                img[selectedJ[:,1], selectedJ[:,0]+1] = [0,255,255]
                                img[selectedJ[:,1], selectedJ[:,0]-1] = [0,255,255]

                                cv2.imwrite("tmp.jpg", img)
                                time.sleep(2)
                        # pdb.set_trace()
                            
                # loss_temp = loss_temp + torch.norm((joints2d[1:] - joints2d[:-1]), dim=1).sum()

                lossFor1Seq = args.cor_loss*loss_cor + args.temp_loss*loss_temp + args.joints2d_loss*loss_2D + args.joints3d_loss*loss_3D
                # lossFor1Seq = args.cor_loss*loss_cor + args.joints2d_loss*loss_2D + args.joints3d_loss*loss_3D
                
                writer.add_scalar('training_loss', lossFor1Seq.item(), iEpoch + iSplit*args.batch_optimization_epochs)
                writer.add_scalar('loss2D', loss_2D.item(), iEpoch + iSplit*args.batch_optimization_epochs)
                writer.add_scalar('loss3D', loss_3D.item(), iEpoch + iSplit*args.batch_optimization_epochs)
                writer.add_scalar('lossTemp', loss_temp.item(), iEpoch + iSplit*args.batch_optimization_epochs)
                writer.add_scalar('lossCor', loss_cor.item(), iEpoch + iSplit*args.batch_optimization_epochs)

                if args.drawImage and (iEpoch % drawImgInterval == 0):
                    frame_gray = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+numImgsInSplit-1), cv2.IMREAD_GRAYSCALE).astype(np.uint8)

                    jointsForVisual = joints2d[-1].detach().cpu().numpy() * 256
                    jointsForVisual = jointsForVisual.astype(np.uint8)                    
                    skeletonOnImg = draw_skeleton(frame_gray, jointsForVisual, draw_edges=True)

                    fig, axs = plt.subplots(1,numImgsInSplit+1,figsize=(20,3))
                    for iImg in range(0, numImgsInSplit+1):
                        if os.path.exists('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iImg)):
                            frame_gray = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iImg), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                        else:
                            print('next image doesnt exist')
                        jointsForVisual = joints2d[iImg].detach().cpu().numpy() * 256
                        jointsForVisual = jointsForVisual.astype(np.uint8)
                        skeletonOnImg = draw_skeleton(frame_gray, jointsForVisual, draw_edges=True)
                        # tmpverts2d_pix = verts2d_pix[iImg].astype(np.uint8)
                        # skeletonOnImg[tmpverts2d_pix[:,1], tmpverts2d_pix[:,0]] = [255, 255, 0]
                        axs[iImg].imshow(skeletonOnImg)
                    fig.canvas.draw()
                    # Convert the canvas to a raw RGB buffer
                    buf = fig.canvas.tostring_rgb()
                    ncols, nrows = fig.canvas.get_width_height()


                    image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
                    # writer.add_images('feat and closest vertics' , featAndVertOnImg, iEpoch*sequence_length + startImg, dataformats='HWC')
                    # writer.add_images('skeleton' , image, iEpoch*sequence_length + startImg, dataformats='HWC')
                    # writer.add_scalar('training_loss', lossFor1Seq.item(), iEpoch*sequence_length + startImg)

                    # writer.add_images('feat and closest vertics' , featAndVertOnImg, iEpoch + iSplit*args.batch_optimization_epochs, dataformats='HWC')
                    writer.add_images('feat and cloest vertices connection' , imgCorFor1Frame/255, iEpoch + iSplit*args.batch_optimization_epochs, dataformats='HWC')
                    writer.add_images('skeleton' , skeletonOnImg, iEpoch + iSplit*args.batch_optimization_epochs, dataformats='HWC')
                    writer.add_images('skeleton in the whole seq' , image, iEpoch + iSplit*args.batch_optimization_epochs, dataformats='HWC')
                    plt.close()
                    writer.add_images('feat and joints temp loss' , imgTempToShow/255, iEpoch + iSplit*args.batch_optimization_epochs, dataformats='HWC')

                    

                lossFor1Seq.backward()
                learnable_params[0].grad
                optimizer.step()
                optimizer.zero_grad()

                if iEpoch % (drawImgInterval*5) == 0:
                    print(iEpoch, f" aver loss for {iSplit}: {lossFor1Seq.item():.3f}")
            
        print('------------------------------------- 3.3. Event-Based Pose Refinement ------------------------------------')
        # learnable_pose_and_shape = torch.load('learnable_parameters-V2.pt')
        # learnable_pose_and_shape.requires_grad_()
        # learnable_params = [learnable_pose_and_shape]
        # Define a hook to freeze a part of the gradient
        # def freeze_gradient(grad):
        #     grad[:, 3:] = 0  # Example: Freeze the first 650 rows
        #     return grad
        # learnable_pose_and_shape.register_hook(freeze_gradient)
        # learnable_params = [learnable_pose_and_shape]
        optimizer = torch.optim.SGD(learnable_params, lr=args.lr_event, momentum=0)
        for iSplit in range(int(args.startWindow),int(args.endWindow)):
            for iEpoch in range(args.event_refinement_epochs):
                print(f"{iEpoch} event_refinement for {iSplit} ")
            
                if (iEpoch % 25 == 24):
                    print(f"{iEpoch} event_refinement for {iSplit} ")
                # print('===== Esil =====')
                loss_sil = torch.tensor(0.0, requires_grad=True)
                startImg = iSplit * numImgsInSplit
                interpolated_rotmats = batch_rodrigues(learnable_pose_and_shape[startImg:(startImg+numImgsInSplit), 3:75].reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
                interpolated_rotmats.shape
                verts, joints3d, _ = model(beta=learnable_pose_and_shape[startImg:(startImg+numImgsInSplit),75:85].view(-1, 10),
                                    theta=None,
                                    get_skin=True,
                                    rotmats=interpolated_rotmats.view(-1, 24, 3, 3))
                verts = verts + learnable_pose_and_shape[startImg:(startImg+numImgsInSplit),:3].unsqueeze(1).repeat(1, verts.shape[1], 1)
                joints3d_trans = joints3d + learnable_pose_and_shape[startImg:(startImg+numImgsInSplit),:3].unsqueeze(1).repeat(1, joints3d.shape[1], 1)
                joints2d = projection_torch(joints3d_trans, cam_intr, H, W)
                verts2d = projection_torch(verts, cam_intr, H, W)

                for iImg in range(numImgsInSplit):
                    verts2dOnSil = vertices_to_silhouette(verts2d[iImg]*H, H, W, device)

                    if torch.isnan(verts2dOnSil).any() or torch.isinf(verts2dOnSil).any():
                        print("Invalid values found in tmp_boundary_pixels")
                        pdb.set_trace()

                    closest_events_u_torch, closest_events_t_torch = find_closest_events_torch_v2(verts2dOnSil,\
                                            events_xy, events_ts, image_raw_event_ts[startImg+iImg], image_raw_event_ts[startImg], \
                                            image_raw_event_ts[startImg+numImgsInSplit], 4, H, W, device)
                    distanceToClosestEvents = torch.norm((verts2dOnSil - closest_events_u_torch)/H, dim = 1)
                    
                    loss_sil = loss_sil + torch.sum((distanceToClosestEvents)) # try sum ( ^2)
                # print('===== Estab =====')
                loss_stab = torch.sum(torch.norm(joints2d - init_joints2d[startImg:(startImg+numImgsInSplit)], dim=2)**2)
                joints2d.shape
                init_joints2d[startImg:(startImg+numImgsInSplit)].shape
                loss_refined = args.sil_loss*loss_sil + args.stab_loss*loss_stab

                loss_refined.backward()
                optimizer.step()
                optimizer.zero_grad()
                learnable_params[0].grad
                learnable_params[0].grad.max()
                # total_loss.append(loss_refined.item())

                writer.add_scalar('refinement_loss', loss_refined.item(), iEpoch + iSplit*args.event_refinement_epochs)
                writer.add_scalar('loss_sil', loss_sil.item(), iEpoch + iSplit*args.event_refinement_epochs)
                writer.add_scalar('loss_stab', loss_stab.item(), iEpoch + iSplit*args.event_refinement_epochs)

                if args.drawImage and (iEpoch % int(drawImgInterval/10) == 1):
                    if os.path.exists('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+numImgsInSplit-1)):
                        frame_gray = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+numImgsInSplit-1), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                    img = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
                    colors = {feature_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for feature_id in range(closest_events_u_torch.shape[0])}
                    startEventIdx = max(img2events[startImg+iImg]-1000, 0)
                    endEventIdx = min(img2events[startImg+iImg]+1000, len(events_xy))
                    events_xys =  events_xy[startEventIdx:endEventIdx].astype(np.int16)
                    img[events_xys[:,1], events_xys[:,0]] = [0,255,0]

                    boundaryAndEventOnImg = draw_feature_dots(img, verts2dOnSil, closest_events_u_torch, [x for x in range(closest_events_u_torch.shape[0])], colors)
                    # verts : black; events: colorful
                    jointsForVisual = joints2d[-1].detach().cpu().numpy() * 256
                    jointsForVisual = jointsForVisual.astype(np.uint8)
                    

                    writer.add_images('boundary_on_Img' , boundaryAndEventOnImg, iEpoch + iSplit*args.event_refinement_epochs, dataformats='HWC')
        print(time.time())
        torch.save(learnable_pose_and_shape, 'learnable_parameters-V2.pt')
        break



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
    parser.add_argument('--batch_optimization_epochs', type=int, default=10000)
    parser.add_argument('--event_refinement_epochs', type=int, default=0)
    parser.add_argument('--joints3d_loss', type=float, default=10) #1
    parser.add_argument('--joints2d_loss', type=float, default=10) #200
    parser.add_argument('--temp_loss', type=float, default=0.01) #80
    parser.add_argument('--cor_loss', type=float, default=2.5) #50
    parser.add_argument('--stab_loss', type=float, default=1) #5
    parser.add_argument('--sil_loss', type=float, default=2)

    parser.add_argument('--startWindow', type=float, default=0) #5
    parser.add_argument('--endWindow', type=float, default=50)

    
    parser.add_argument('--lr_start', '-lr', type=float, default=0.001)
    parser.add_argument('--lr_event', '-lre', type=float, default=0.001)

    parser.add_argument('--drawImage', type=bool, default=False)

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
