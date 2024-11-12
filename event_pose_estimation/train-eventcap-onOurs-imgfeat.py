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
    vertices_to_silhouette_Extreme, draw_feature_dots_lines, txt_to_np, find_closest_events_torch_v2
from event_pose_estimation.geometry import projection_torch, rot6d_to_rotmat, delta_rotmat_to_rotmat
from event_pose_estimation.loss_funcs import compute_mpjpe, compute_pa_mpjpe, compute_pa_mpjpe_eventcap
import h5py
import random
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
from PIL import Image
import io

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
    feature_pts = txt_to_np("/home/ziyan/02_research/EventHPE/tracks.txt")




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

    R = np.asarray(h5['annotations']['R'])[::8]
    T = np.asarray(h5['annotations']['T'])[::8]
    poses = np.asarray(h5['annotations']['poses'])[::8]
    shape = np.asarray(h5['annotations']['shape'])[::8]
    
    event_intr = np.asarray(h5['calibration']['event_intr'])
    event_extr = np.asarray(h5['calibration']['event_extr'])
    image_intr = np.asarray(h5['calibration']['image_intr'])
    image_extr = np.asarray(h5['calibration']['image_extr'])

    img2events = np.searchsorted(t, real_image_ts)

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



    # num_interpolations = 10
    numImgsInSplit = 8
    drawImgInterval = 100
    

    sequence_length = len(image_data)
    # set model
    modelSeq = SMPL(smpl_dir, sequence_length)
    # Freeze the rest of the SMPL model parameters
    for param in modelSeq.parameters():
        param.requires_grad = False  # Freeze all SMPL parameters

    modelSeq = modelSeq.to(device=device)  # move the model parameters to CPU/GPU

    totalSplits = sequence_length // numImgsInSplit
    # with h5py.File('/home/ziyan/02_research/EventHPE/events.hdf5', 'r') as f:
    #     # ['events_p', 'events_t', 'events_xy', 'image_raw_event_ts']
    #     image_raw_event_ts = np.array(f['image_raw_event_ts'])
    #     events_xy = np.concatenate([np.array(f['x'])[:,np.newaxis], np.array(f['y'])[:,np.newaxis]], axis=1)
    #     events_ts = np.array(f['t'])
    #     img2events = np.searchsorted(events_ts, image_raw_event_ts)
    
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
        _tran0, _pose0, _shape0 = T[startImg], poses[startImg], shape[startImg]
        _tranN, _poseN, _shapeN = T[startImg+numImgsInSplit], poses[startImg+numImgsInSplit], shape[startImg+numImgsInSplit]

        alphas = np.linspace(1 / numImgsInSplit, (numImgsInSplit-1) / numImgsInSplit, (numImgsInSplit-1))
        _params0 = np.concatenate([_tran0, _pose0, _shape0], axis=0)
        _paramsN = np.concatenate([_tranN, _poseN, _shapeN], axis=0)
        _paramsF = (1 - alphas[:, np.newaxis]) * _params0 + alphas[:, np.newaxis] * _paramsN

        learnable_pose_and_shape[startImg:(startImg+numImgsInSplit),:] = \
            torch.from_numpy(np.concatenate([_params0[np.newaxis,:], _paramsF], axis=0))


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

    cam_intr = event_intr[1:]
    cam_intr[:2] *= 0.6 
    cam_intr = torch.from_numpy(cam_intr).float()
    Height, Width = 640, 480
    

    



    rot_h = batch_rodrigues(torch.tensor(R[:joints3d.shape[0]], device=device).reshape(-1, 3))
    R_extr = torch.tensor(event_extr[:3,:3], device = device)
    T_extr = torch.tensor(event_extr[:3,-1], device = device)
    joints3d_trans = torch.matmul(joints3d, rot_h.transpose(1,2)) + learnable_pose_and_shape[:joints3d.shape[0], :3].unsqueeze(1).repeat(1, joints3d.shape[1], 1)
    joints3d_trans = torch.matmul(joints3d_trans, R_extr.T) + T_extr 

    # joints3d_trans = joints3d + learnable_pose_and_shape[:, :3].unsqueeze(1).repeat(1, joints3d.shape[1], 1)
    joints2d = projection_torch(joints3d_trans, cam_intr, Height, Width)
    init_joints2d = joints2d.clone().detach()

    # get joints 2d and 3d from HMR for E2d and E3d
    # get transHMR from every frame
    transHMREveryFrame = torch.zeros(totalSplits*numImgsInSplit,3).to(device)
    for iSplit in range(totalSplits):
        joint2DHMR, joint3DHMR, transHMR = \
            joints2d[iSplit*numImgsInSplit].detach(), joints3d_trans[iSplit*numImgsInSplit].detach(), T[iSplit*numImgsInSplit]
        joints2dFromHMR[iSplit] = joint2DHMR
        joints3dFromHMR[iSplit] = joint3DHMR
        transHMREveryFrame[iSplit*numImgsInSplit: (iSplit+1)*numImgsInSplit] = \
            torch.from_numpy(T[iSplit*numImgsInSplit: (iSplit+1)*numImgsInSplit]).to(device)

    # set model
    model = SMPL(smpl_dir, numImgsInSplit+1)
    # Freeze the rest of the SMPL model parameters
    for param in model.parameters():
        param.requires_grad = False  # Freeze all SMPL parameters

    model = model.to(device=device)  # move the model parameters to CPU/GPU


    print('===== Image Feature Extraction =====')
    featsEveryFrame = []
    featOnEveryBackward = []
    featOnEveryForward = []

    for iSplit in range(totalSplits):
        
        # Forward Feature Extraction and backward feature extraction 
        # Forward: Take first frame and find corners in it; Backward: Take the last frame
        startImg = iSplit * numImgsInSplit
        
        img = Image.open(io.BytesIO(image_data[startImg]))
        img = np.array(img)
        image_warped = cv2.warpPerspective(np.array(img),H,dsize=(480,640))

        first_grey = cv2.cvtColor(image_warped, cv2.COLOR_BGR2GRAY) 
        imgsIn1Split = first_grey[np.newaxis, :,:]

        for iImg in range(1, numImgsInSplit+1):
            img = Image.open(io.BytesIO(image_data[startImg+iImg]))
            img = np.array(img)
            image_warped = cv2.warpPerspective(np.array(img),H,dsize=(480,640))
            frame_grey = cv2.cvtColor(image_warped, cv2.COLOR_BGR2GRAY) 
            imgsIn1Split = np.vstack([imgsIn1Split, frame_grey[np.newaxis, :,:]])
        featsForward = findImgFeat(imgsIn1Split) # 0->8
        featsBackward = findImgFeat(imgsIn1Split[::-1]) # 8->0

        # stitching the bidirectional features
        featsBackwardReversed = featsBackward[::-1] 
        for iImg in range(1,len(featsForward)-1):
            
            distances = distance_matrix(featsForward[iImg], featsBackwardReversed[iImg])
            # Apply the Hungarian algorithm to find the optimal one-to-one correspondence
            row_indices, col_indices = linear_sum_assignment(distances)

            matched_forward = featsForward[iImg][row_indices]
            matched_backward = featsBackwardReversed[iImg][col_indices]

            matched_mean = (matched_forward + matched_backward)/2
            stitch_mask = np.linalg.norm(matched_forward - matched_backward, axis=1) < 4
            if (0):
                tmpP = matched_mean.astype(np.uint16)
                tmpPF = matched_forward.astype(np.uint16)
                tmpPB = matched_backward.astype(np.uint16)
                img = Image.open(io.BytesIO(image_data[startImg+iImg]))
                img = np.array(img)
                image_warped = cv2.warpPerspective(np.array(img),H,dsize=(480,640))
                frame_gray = cv2.cvtColor(image_warped, cv2.COLOR_BGR2GRAY) 
                img = draw_feature_dots_lines(frame_gray, tmpPF[stitch_mask], tmpPB[stitch_mask], np.arange(len(tmpP)), Height, Width)
                cv2.imwrite("tmp.jpg", img)
                time.sleep(0.5)

            if (0):
                tmpP = matched_mean.astype(np.uint16)
                tmpPF = matched_forward.astype(np.uint16)
                tmpPB = matched_backward.astype(np.uint16)
                img = Image.open(io.BytesIO(image_data[startImg+iImg]))
                img = np.array(img)
                image_warped = cv2.warpPerspective(np.array(img),H,dsize=(480,640))
                frame_gray = cv2.cvtColor(image_warped, cv2.COLOR_BGR2GRAY) 
                img = draw_feature_dots_lines(frame_gray, tmpPF[stitch_mask], tmpPB[stitch_mask], np.arange(len(tmpP)), Height, Width)
                cv2.imwrite("tmp.jpg", img)
                
                tmpPB.shape
                tmpPF.shape

                img = Image.open(io.BytesIO(image_data[startImg+iImg]))
                img = np.array(img)
                image_warped = cv2.warpPerspective(np.array(img),H,dsize=(480,640))
                image_warped.shape
                frame_gray = cv2.cvtColor(image_warped, cv2.COLOR_BGR2GRAY) 
                frame_gray.shape
                colors = {feature_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for feature_id in range(len(tmpP))}
                img = draw_feature_dots_lines(imgsIn1Split[iImg-1], tmpP, tmpP, np.arange(len(tmpP)), colors)
                img = draw_feature_dots_lines(frame_gray, tmpP, tmpP, np.arange(len(tmpP)), Height, Width)
                img = draw_feature_dots(imgsIn1Split[iImg], featsForward[iImg-1], tmpPF, np.arange(len(tmpP)), colors)
                img = draw_feature_dots(imgsIn1Split[iImg], featsForward[iImg+1], tmpPF, np.arange(len(tmpP)), colors)
                img = draw_feature_dots_lines(frame_gray, tmpPF[stitch_mask], tmpPB[stitch_mask], np.arange(len(tmpP)), Height, Width)
                img = draw_feature_dots_lines(frame_gray, tmpPB, tmpPB, np.arange(len(tmpP)), colors)
                cv2.imwrite("tmp.jpg", img)
            featsEveryFrame.append(matched_mean[stitch_mask])
            featOnEveryForward.append(featsForward[iImg-1][row_indices][stitch_mask])
            featOnEveryBackward.append(featsBackwardReversed[iImg+1][col_indices][stitch_mask])

    
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
            rotmats = batch_rodrigues(learnable_pose_and_shape[startImg:(startImg+numImgsInSplit+1), 3:75]\
                                        .reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
            verts, joints3d, _ = model(beta=learnable_pose_and_shape[startImg:(startImg+numImgsInSplit+1),75:85].view(-1, 10),
                                theta=None,
                                get_skin=True,
                                rotmats=rotmats.view(-1, 24, 3, 3))
            
            rot_h = batch_rodrigues(torch.tensor(R[startImg:(startImg+numImgsInSplit+1)], device=device).reshape(-1, 3))
            R_extr = torch.tensor(event_extr[:3,:3], device = device)
            T_extr = torch.tensor(event_extr[:3,-1], device = device)

            # R_root, T_root = model.get_root_rt(learnable_pose_and_shape[startImg:(startImg+numImgsInSplit+1),75:85], learnable_pose_and_shape[startImg:(startImg+numImgsInSplit+1), 3:75])
            verts = torch.matmul(verts, rot_h.transpose(1,2)) + learnable_pose_and_shape[startImg:(startImg+numImgsInSplit+1), :3]\
                                                                .unsqueeze(1).repeat(1, verts.shape[1], 1)
            verts = torch.matmul(verts, R_extr.T) + T_extr 

            joints3d_trans = torch.matmul(joints3d, rot_h.transpose(1,2)) + learnable_pose_and_shape[startImg:(startImg+numImgsInSplit+1), :3]\
                                                                            .unsqueeze(1).repeat(1, joints3d.shape[1], 1)
            joints3d_trans = torch.matmul(joints3d_trans, R_extr.T) + T_extr 

            joints2d = projection_torch(joints3d_trans, cam_intr, Height, Width)
            verts2d = projection_torch(verts, cam_intr, Height, Width)
            # pdb.set_trace()
            if (0):
                for i in range(verts.shape[0]):

                    tmpV = np.clip((verts2d[i].detach().cpu().numpy() * np.array([Width, Height])), 0, [Width-1, Height-1]).astype(np.uint16)
                    tmpJoints2d = np.clip((joints2d[i].detach().cpu().numpy() * np.array([Width, Height])), 0, [Width-1, Height-1]).astype(np.uint16)
                    img = Image.open(io.BytesIO(image_data[startImg+i]))
                    img = np.array(img)
                    image_warped = cv2.warpPerspective(np.array(img),H,dsize=(480,640))
                    frame_gray = cv2.cvtColor(image_warped, cv2.COLOR_BGR2GRAY) 
                    frame_gray.shape
                    tmpV[:,1].max()
                    tmpV[:,0].max()
                    # frame_gray[tmpV[:,1], tmpV[:,0]] = [255]
                    frame_gray[tmpJoints2d[:,1], tmpJoints2d[:,0]] = [255]
                    cv2.imwrite("tmp.jpg", frame_gray)
                    time.sleep(1)

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
            tolerance = 3
            startFeatFrame = iSplit*(numImgsInSplit-1)
            imgCorFor1Frame = np.zeros((Height*(numImgsInSplit-1), Width*3, 3))

            for iP in range(0, numImgsInSplit-1):
                totalSplits * numImgsInSplit *7 /8
                pCurr = torch.tensor(featsEveryFrame[startFeatFrame + iP], requires_grad=False).to(device)  # (n, 2)
                rough_constraint_for_feat = init_joints2d[startImg + iP+1] * torch.tensor([Width, Height]).to(device)
                distances = torch.norm(pCurr.unsqueeze(1) - rough_constraint_for_feat.unsqueeze(0), dim=2)  # Resulting shape (N, M)                
                feature_around_body_mask = torch.min(distances,dim=1)[0] < 30

                verts2dPix = verts2d[iP+1]*torch.tensor([Width, Height],device=device)
                min_distances, closest_vert_indices = findClosestPointTorch(verts2dPix, pCurr[feature_around_body_mask])
                
                # tau(i,h) filter out the non close feature
                valid_mask = min_distances < tolerance

                # the same feat in the next frame and the same closest vertics in the next frame
                pNext = torch.tensor(featOnEveryBackward[startFeatFrame + iP], requires_grad=False).to(device)[feature_around_body_mask][valid_mask]  # (n, 2)
                vNext = verts2d[iP+2,closest_vert_indices][valid_mask]*torch.tensor([Width, Height],device=device)
                nextDistances = pNext - vNext

                # the same feat in the last frame and the same closest vertics in the last frame
                pLast = torch.tensor(featOnEveryForward[startFeatFrame + iP], requires_grad=False).to(device)[feature_around_body_mask][valid_mask]  # (n, 2)
                vLast = verts2d[iP,closest_vert_indices][valid_mask]*torch.tensor([Width, Height],device=device)
                lastDistances = pLast - vLast

                if args.drawImage and (iEpoch % drawImgInterval == 0):
                # if args.drawImage:
                    # tmpP = pCurr[valid_mask].clone().detach().cpu().numpy().astype(np.uint8)
                    # tmpV = (verts2d[iP+1]*H)[closest_vert_indices][valid_mask].detach().cpu().numpy().astype(np.uint8)
                    tmpP = pCurr[feature_around_body_mask][valid_mask].clone().detach().cpu().numpy()
                    tmpV = (verts2d[iP+1]*torch.tensor([Width, Height],device=device))[closest_vert_indices][valid_mask].detach().cpu().numpy()
                    img = Image.open(io.BytesIO(image_data[startImg+iP+1]))
                    img = np.array(img)
                    image_warped = cv2.warpPerspective(np.array(img),H,dsize=(480,640))
                    tmpVAll = np.clip((verts2d[iP+1]*torch.tensor([Width, Height],device=device)).detach().cpu().numpy(), 0,\
                                       [Width-1, Height-1]).astype(np.uint16)
                    image_warped[tmpVAll[:,1], tmpVAll[:,0]] = [0,0,255]
                    # frame_gray = cv2.cvtColor(image_warped, cv2.COLOR_BGR2GRAY) 
                    img = draw_feature_dots_lines(image_warped, tmpP, tmpV, np.arange(len(tmpP)), Height, Width)
                    # feat: [255,255,0]; vert: [0,255,255]
                    imgCorFor1Frame[ iP*Height: (iP+1)*Height, Width:Width*2, :] = img
                    # cv2.imwrite("tmp.jpg", imgCorFor1Frame)

                    # tmpNextP = pNext.clone().detach().cpu().numpy().astype(np.uint8)
                    # tmpNextV = (verts2d[iP+2]*H)[closest_vert_indices][valid_mask].detach().cpu().numpy().astype(np.uint8)
                    tmpNextP = pNext.clone().detach().cpu().numpy()
                    tmpNextV = (verts2d[iP+2]*torch.tensor([Width, Height],device=device))[closest_vert_indices][valid_mask].detach().cpu().numpy()
                    img = Image.open(io.BytesIO(image_data[startImg+iP+2]))
                    img = np.array(img)
                    image_warped = cv2.warpPerspective(np.array(img),H,dsize=(480,640))
                    tmpVAll = np.clip((verts2d[iP+2]*torch.tensor([Width, Height],device=device)).detach().cpu().numpy(), 0,\
                                       [Width-1, Height-1]).astype(np.uint16)
                    image_warped[tmpVAll[:,1], tmpVAll[:,0]] = [0,0,255]
                    # frame_grayNext = cv2.cvtColor(image_warped, cv2.COLOR_BGR2GRAY) 
                    # colors = {feature_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for feature_id in range(len(tmpNextP))}

                    img = draw_feature_dots_lines(image_warped, tmpNextP, tmpNextV, np.arange(len(tmpNextP)), Height, Width)
                    imgCorFor1Frame[ iP*Height: (iP+1)*Height, Width*2:Width*3, :] = img
                    # cv2.imwrite("tmp.jpg", img)

                    tmpLastP = pLast.clone().detach().cpu().numpy().astype(np.uint8)
                    tmpLastV = (verts2d[iP]*torch.tensor([Width, Height],device=device))[closest_vert_indices][valid_mask].detach().cpu().numpy().astype(np.uint8)
                    tmpLastP = pLast.clone().detach().cpu().numpy()
                    tmpLastV = (verts2d[iP]*torch.tensor([Width, Height],device=device))[closest_vert_indices][valid_mask].detach().cpu().numpy()
                    tmpVAll = np.clip((verts2d[iP]*torch.tensor([Width, Height],device=device)).detach().cpu().numpy(), 0,[Width-1, Height-1]).astype(np.uint16)
                
                    img = Image.open(io.BytesIO(image_data[startImg+iP]))
                    img = np.array(img)
                    image_warped = cv2.warpPerspective(np.array(img),H,dsize=(480,640))
                    tmpVAll = np.clip((verts2d[iP]*torch.tensor([Width, Height],device=device)).detach().cpu().numpy(), 0,\
                                       [Width-1, Height-1]).astype(np.uint16)
                    image_warped[tmpVAll[:,1], tmpVAll[:,0]] = [0,0,255]
                    # frame_grayLast = cv2.cvtColor(image_warped, cv2.COLOR_BGR2GRAY) 
                    # colors = {feature_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for feature_id in range(len(tmpLastP))}
                    img = draw_feature_dots_lines(image_warped, tmpLastP, tmpLastV, np.arange(len(tmpLastP)), Height, Width)
                    # img[tmpVAll[:,1], tmpVAll[:,0]] = [0,0,255]
                    imgCorFor1Frame[ iP*Height: (iP+1)*Height, :Width, :] = img
                    
                    # pdb.set_trace()



                    if (0):
                        featOnEveryForward
                        pLast = torch.tensor(featOnEveryForward[startFeatFrame], requires_grad=False).to(device)[valid_mask]  # (n, 2)
                        vLast = verts2d[iP,closest_vert_indices][valid_mask]*torch.tensor([Width, Height],device=device)
                        lastDistances = pLast - vLast
                        verts2d[0,:,1].max()
                        verts2d[4,:,1].max()

                totalDistances = nextDistances + lastDistances
                selectedDistancesSqSum = torch.sum(torch.norm(totalDistances*torch.tensor([1/Width, 1/Height], device=device),\
                                                                dim=1)**2)
                loss_cor = loss_cor + selectedDistancesSqSum
            # cv2.imwrite("tmp.jpg", imgCorFor1Frame)
            # pdb.set_trace()

            # print('===== Etemp =====')
            loss_temp = torch.tensor(0.0, requires_grad=True)
            if (1):
                toleranceTemp = 20
                imgTempToShow = np.zeros((Height, Width*(numImgsInSplit+1), 3))
                for iImg in range(0, numImgsInSplit):
                    if iImg == 0:
                        p0 = torch.tensor(featOnEveryForward[startFeatFrame], requires_grad=False).to(device)  # (n, 2)
                    elif iImg == numImgsInSplit:
                        p0 = torch.tensor(featOnEveryBackward[startFeatFrame+numImgsInSplit-2], requires_grad=False).to(device)  # (n, 2)
                    else:
                        p0 = torch.tensor(featsEveryFrame[startFeatFrame+iImg-1], requires_grad=False).to(device)  # (n, 2)
                    
                    events_xys = xy_u.T[img2events[startImg+iImg]-1000:img2events[startImg+iImg]+1000]
                    events_xys = np.clip(events_xys, 0, [Width-1, Height-1]).astype(np.uint16)

                    
                    joints2dPix = joints2d[iImg] * torch.tensor([Width, Height], device=device)
                    # calculate distance to each feature points
                    distances = torch.norm(joints2dPix.unsqueeze(1) - p0.unsqueeze(0), dim=2)  # Resulting shape (N, M)
                    selectedJoints = torch.min(distances, dim=1)[0] > toleranceTemp
                    joints3d[iImg][selectedJoints].shape
                    # min_distances, closest_joints_indices = findClosestPointTorch(joints2dPix, p0)
                    # selectedJoints = torch.unique(closest_joints_indices[min_distances > toleranceTemp])
                    
                    loss_temp = loss_temp + torch.norm((joints3d[iImg][selectedJoints] - joints3d[iImg+1][selectedJoints]), dim=1).sum()
                    
                    if args.drawImage and (iEpoch % drawImgInterval == 0):
                        img = Image.open(io.BytesIO(image_data[startImg+iImg]))
                        img = np.array(img)
                        image_warped = cv2.warpPerspective(np.array(img),H,dsize=(480,640))
                        tmpJ = joints2dPix[selectedJoints].detach().cpu().numpy().astype(np.uint16)
                        tmpJAll = joints2dPix.detach().cpu().numpy().astype(np.uint16)
                        tmpP = p0.detach().cpu().numpy().astype(np.uint16)
                        for iJ in range(len(tmpP)):
                            cv2.circle(image_warped, (tmpP[iJ,0], tmpP[iJ,1]), 3, [0,255,0], -1)
                        for iJ in range(len(tmpJAll)):
                            cv2.circle(image_warped, (tmpJAll[iJ,0], tmpJAll[iJ,1]), 3, [255,0,0], -1)
                        for iJ in range(len(tmpJ)):
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.3  # Size of the text
                            selectedID = np.array([range(len(joints3d[iImg]))]).T[selectedJoints.detach().cpu().numpy() == True]
                            selectedID[0]
                            cv2.putText(image_warped, str(selectedID[iJ].item()), (tmpJ[iJ,0]+3, tmpJ[iJ,1]+3), font, font_scale, [0,0,255], 1)
                            cv2.circle(image_warped, (tmpJ[iJ,0], tmpJ[iJ,1]), 3, [0,0,255], -1)
                        imgTempToShow[:, iImg*Width: (iImg+1)*Width, :] = image_warped
                    # cv2.imwrite("tmp.jpg", imgTempToShow)
                    # pdb.set_trace()
            # loss_temp = loss_temp + torch.norm((joints2d[1:] - joints2d[:-1]), dim=2).sum()


            lossFor1Seq = args.cor_loss*loss_cor + args.temp_loss*loss_temp + args.joints2d_loss*loss_2D + args.joints3d_loss*loss_3D
            writer.add_scalar('training_loss', lossFor1Seq.item(), iEpoch + iSplit*args.batch_optimization_epochs)
            writer.add_scalar('loss2D', loss_2D.item(), iEpoch + iSplit*args.batch_optimization_epochs)
            writer.add_scalar('loss3D', loss_3D.item(), iEpoch + iSplit*args.batch_optimization_epochs)
            writer.add_scalar('lossTemp', loss_temp.item(), iEpoch + iSplit*args.batch_optimization_epochs)
            writer.add_scalar('lossCor', loss_cor.item(), iEpoch + iSplit*args.batch_optimization_epochs)

            if lossFor1Seq.item() > 10000:
                pdb.set_trace()
            # lossFor1Seq = args.cor_loss*loss_cor + args.joints2d_loss*loss_2D + args.joints3d_loss*loss_3D
            
            if args.drawImage and (iEpoch % drawImgInterval == 0):
            # if (0):
                print("loss 3d: ", loss_3D.item())
                print("loss temp: ", loss_temp.item())
                fig, axs = plt.subplots(1,9,figsize=(20,3))
                for iImg in range(0, numImgsInSplit+1):
                    jointsForVisual = joints2d[iImg].detach().cpu().numpy()
                    jointsForVisual[:,0], jointsForVisual[:,1] = jointsForVisual[:,0]*Width, jointsForVisual[:,1]*Height
                    jointsForVisual = jointsForVisual.astype(np.uint16)
                    


                    
                    img = Image.open(io.BytesIO(image_data[startImg+iImg]))
                    img = np.array(img)
                    image_warped = cv2.warpPerspective(np.array(img),H,dsize=(480,640))

                    frame_gray = image_warped 

                    skeletonOnImg = draw_skeleton(frame_gray, jointsForVisual, draw_edges=True)
                    
                    
                    if (iImg == 0):
                        first2DHMR = joints2dFromHMR[iSplit].detach().cpu().numpy()
                        first2DHMR = first2DHMR * np.array([Width, Height])
                        first2DHMR = first2DHMR.astype(np.uint16)
                        skeletonOnImg = draw_skeleton(skeletonOnImg, first2DHMR, draw_edges=True)

                    if (iImg == numImgsInSplit):
                        end2DHMR = joints2dFromHMR[iSplit+1].detach().cpu().numpy()
                        end2DHMR = end2DHMR * np.array([Width, Height])
                        end2DHMR = end2DHMR.astype(np.uint16)
                        skeletonOnImg = draw_skeleton(skeletonOnImg, end2DHMR, draw_edges=True)
                        
                        
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
                writer.add_images('feat and joints temp loss' , imgTempToShow/255, iEpoch + iSplit*args.batch_optimization_epochs, dataformats='HWC')

                # writer.add_images('skeleton' , skeletonOnImg, iEpoch + iSplit*args.batch_optimization_epochs, dataformats='HWC')
                writer.add_images('skeleton in the whole seq' , image, iEpoch + iSplit*args.batch_optimization_epochs, dataformats='HWC')
                plt.close()
                
                

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

    optimizer = torch.optim.SGD(learnable_params, lr=args.lr_event, momentum=0)
    for iSplit in range(int(args.startWindow),int(args.endWindow)):
        for iEpoch in range(args.event_refinement_epochs):
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



            rot_h = batch_rodrigues(torch.tensor(R[startImg:(startImg+numImgsInSplit)], device=device).reshape(-1, 3))
            R_extr = torch.tensor(event_extr[:3,:3], device = device)
            T_extr = torch.tensor(event_extr[:3,-1], device = device)

            # R_root, T_root = model.get_root_rt(learnable_pose_and_shape[startImg:(startImg+numImgsInSplit+1),75:85], learnable_pose_and_shape[startImg:(startImg+numImgsInSplit+1), 3:75])
            verts = torch.matmul(verts, rot_h.transpose(1,2)) + learnable_pose_and_shape[startImg:(startImg+numImgsInSplit), :3]\
                                                                .unsqueeze(1).repeat(1, verts.shape[1], 1)
            verts = torch.matmul(verts, R_extr.T) + T_extr 
            joints3d_trans = torch.matmul(joints3d, R_extr.T) + T_extr 
            
            joints3d_trans = torch.matmul(joints3d_trans, rot_h.transpose(1,2)) + learnable_pose_and_shape[startImg:(startImg+numImgsInSplit), :3]\
                                                                .unsqueeze(1).repeat(1, joints3d_trans.shape[1], 1)
            joints2d = projection_torch(joints3d_trans, cam_intr, Height, Width)
            verts2d = projection_torch(verts, cam_intr, Height, Width)

            for iImg in range(numImgsInSplit):
                verts2dOnSil = vertices_to_silhouette_Extreme(verts2d[iImg]*torch.tensor([Width, Height], device=device), Height, Width, device)
                
                # events_xys = np.concatenate([y[img2events[startImg]:img2events[startImg+numImgsInSplit]][:,np.newaxis],\
                #                            x[img2events[startImg]:img2events[startImg+numImgsInSplit]][:,np.newaxis]],axis=1)
                events_xys = xy_u.T[img2events[startImg]:img2events[startImg+numImgsInSplit]]
                # tmpE = np.clip(events_xys, 0, [Width-1, Height-1]).astype(np.uint16)
                # img = Image.open(io.BytesIO(image_data[startImg+numImgsInSplit]))
                # img = np.array(img)
                # image_warped = cv2.warpPerspective(np.array(img),H,dsize=(480,640))
                # image_warped[tmpE[:,1], tmpE[:,0]] = [0,0,255]


                events_ts = t[img2events[startImg]:img2events[startImg+numImgsInSplit]]
                if (0):
                    tmpS = verts2dOnSil.detach().cpu().numpy().astype(np.uint16)
                    tmpS.shape
                    img = np.zeros((Height, Width))
                    img[tmpS[:,1], tmpS[:,0]] = [255]
                    cv2.imwrite("tmp.jpg", image_warped)
                closest_events_u_torch, closest_events_t_torch = find_closest_events_torch_v2(verts2dOnSil,\
                                        events_xys, events_ts, event_trigger[(startImg+iImg)*8], event_trigger[startImg], \
                                        event_trigger[startImg+numImgsInSplit], args.dist, Height, Width, device)
                distanceToClosestEvents = torch.norm((verts2dOnSil - closest_events_u_torch)*torch.tensor([1/Width, 1/Height], device=device), dim = 1)
                loss_sil = loss_sil + torch.mean((distanceToClosestEvents)) # try sum ( ^2)
            
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
            print(f"{iEpoch} event_refinement for {iSplit} : {loss_refined.item()}")


            # if args.drawImage and (iEpoch % int(drawImgInterval/10) == 1):
            if args.drawImage:
                img = Image.open(io.BytesIO(image_data[startImg+numImgsInSplit-1]))
                img = np.array(img)
                image_warped = cv2.warpPerspective(np.array(img),H,dsize=(480,640))
                frame_gray = image_warped 
                # events_xys = np.concatenate([y[img2events[startImg+numImgsInSplit]-10000:img2events[startImg+numImgsInSplit]+10000][:,np.newaxis],\
                #                            x[img2events[startImg+numImgsInSplit]-10000:img2events[startImg+numImgsInSplit]+10000][:,np.newaxis]],axis=1)
                events_xys = xy_u.T[img2events[startImg+numImgsInSplit]-3000:img2events[startImg+numImgsInSplit]]
                events_xys = np.clip(events_xys, 0, [Width-1, Height-1]).astype(np.uint16)
                frame_gray[events_xys[:,1], events_xys[:,0]] = [100,100,100]
                colors = {feature_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for feature_id in range(closest_events_u_torch.shape[0])}
                if (0):
                    tmpT = t[img2events[startImg+numImgsInSplit]-3000:img2events[startImg+numImgsInSplit]]
                    verts2dOnSil.shape
                    tmpImg = frame_gray.copy()
                    tmpB = verts2dOnSil.detach().cpu().numpy().astype(np.uint16)
                    closest_event = closest_events_u_torch.detach().cpu().numpy().astype(np.uint16)
                    tmpImg[tmpB[:,1], tmpB[:,0]] = [0,255,255]
                    tmpImg[closest_event[:,1], closest_event[:,0]] = [0,0,255]
                    tmpImg[closest_event[:,1]+1, closest_event[:,0]] = [0,0,255]
                    tmpImg[closest_event[:,1], closest_event[:,0]+1] = [0,0,255]
                    tmpImg[closest_event[:,1]-1, closest_event[:,0]] = [0,0,255]
                    tmpImg[closest_event[:,1], closest_event[:,0]-1] = [0,0,255]
                    cv2.imwrite('tmp.jpg', tmpImg)

                boundaryAndEventOnImg = draw_feature_dots(frame_gray, verts2dOnSil, closest_events_u_torch, [x for x in range(closest_events_u_torch.shape[0])], colors)
                # verts : black; events: colorful
                jointsForVisual = joints2d[-1].detach().cpu().numpy() * 256
                jointsForVisual = jointsForVisual.astype(np.uint8)
                
                # skeletonOnImg = draw_skeleton(boundaryAndEventOnImg, jointsForVisual, draw_edges=True)

                writer.add_images('boundary_on_Img' , boundaryAndEventOnImg, iEpoch + iSplit*args.event_refinement_epochs, dataformats='HWC')
    print(time.time())
    torch.save(learnable_pose_and_shape, 'learnable_parameters-onOurs.pt')



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
    parser.add_argument('--batch_optimization_epochs', type=int, default=10000)
    parser.add_argument('--event_refinement_epochs', type=int, default=100)
    parser.add_argument('--joints3d_loss', type=float, default=10) #1
    parser.add_argument('--joints2d_loss', type=float, default=5) #200
    parser.add_argument('--temp_loss', type=float, default=0.05) #80
    parser.add_argument('--cor_loss', type=float, default=0.03) #50
    parser.add_argument('--stab_loss', type=float, default=0.1)
    parser.add_argument('--sil_loss', type=float, default=20)

    parser.add_argument('--dist', type=float, default=4)


    parser.add_argument('--startWindow', type=float, default=5) #5
    parser.add_argument('--endWindow', type=float, default=6)

    
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
