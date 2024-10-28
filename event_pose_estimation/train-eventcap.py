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
    joints2dandFeatOnImage, draw_feature_dots, draw_skeleton
from event_pose_estimation.geometry import projection_torch, rot6d_to_rotmat, delta_rotmat_to_rotmat
import h5py
import random

# import event_pose_estimation.utils as util



def train(args):
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu_id if use_cuda else "cpu")
    dtype = torch.float32

    # # set dataset
    dataset_train = TrackingDataloader(
        data_dir=args.data_dir,
        max_steps=args.max_steps,
        num_steps=args.num_steps,
        skip=args.skip,
        events_input_channel=args.events_input_channel,
        img_size=args.img_size,
        mode='train',
        use_flow=args.use_flow,
        use_flow_rgb=args.use_flow_rgb,
        use_hmr_feats=args.use_hmr_feats,
        use_vibe_init=args.use_vibe_init,
        use_hmr_init=args.use_hmr_init,
    )
    # train_generator = DataLoader(
    #     dataset_train,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_worker,
    #     pin_memory=args.pin_memory
    # )
    # total_iters = len(dataset_train) // args.batch_size + 1

    # dataset_val = TrackingDataloader(
    #     data_dir=args.data_dir,
    #     max_steps=args.max_steps,
    #     num_steps=args.num_steps,
    #     skip=args.skip,
    #     events_input_channel=args.events_input_channel,
    #     img_size=args.img_size,
    #     mode='test',
    #     use_flow=args.use_flow,
    #     use_flow_rgb=args.use_flow_rgb,
    #     use_hmr_feats=args.use_hmr_feats,
    #     use_vibe_init=args.use_vibe_init,
    #     use_hmr_init=args.use_hmr_init,
    # )
    # val_generator = DataLoader(
    #     dataset_val,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_worker,
    #     pin_memory=args.pin_memory
    # )

    smpl_dir = '/home/ziyan/02_research/EventHPE/smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
    print('[smpl_dir] %s' % smpl_dir)

    mse_func = torch.nn.MSELoss()
    if args.use_amp: # true
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None


    # set tensorboard
    start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    writer = SummaryWriter('%s/%s/%s' % (args.result_dir, args.log_dir, start_time))
    print('[tensorboard] %s/%s/%s' % (args.result_dir, args.log_dir, start_time))
    if args.model_dir is not None:
        print('[model dir] model loaded from %s' % args.model_dir)
        checkpoint = torch.load(args.model_dir, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    save_dir = '%s/%s/%s' % (args.result_dir, args.log_dir, start_time)
    model_dir = '%s/%s/%s/model_events_pose.pkl' % (args.result_dir, args.log_dir, start_time)

    # training
    best_loss = 1e4
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
        
        print('===== E2D and E3D =====')
        # set initial SMPL parameters 
        # E2D implementation has ignored the optimization on OpenPose, just copy hmr result and interpolate to every tracking frame
        # Set initial SMPL parameters as learnable (e.g., pose and shape)
        learnable_pose_and_shape = torch.randn(totalSplits*numImgsInSplit, 85, device=device)
        interpolated_ori_trans = torch.zeros(totalSplits*numImgsInSplit,3).to(device)
        init_joints2d = torch.zeros(totalSplits*numImgsInSplit, 24, 2)
        
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
        optimizer = torch.optim.Adam(learnable_params, lr=args.lr_start)
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
        cam_intr = torch.from_numpy(dataset_train.cam_intr).float()
        H, W = 256, 256
        interpolated_ori_trans.shape
        joints3d_trans = joints3d + interpolated_ori_trans.unsqueeze(1).repeat(1, joints3d.shape[1], 1)
        joints2d = projection_torch(joints3d_trans, cam_intr, H, W)
        joints2d.shape
        init_joints2d = joints2d.clone().detach()
        # joints2dOnImage(joints2d, learnable_pose_and_shape.shape[0], args, action, saved_folder = "/home/ziyan/02_research/EventHPE/event_pose_estimation/init_folder")

    
        # set model
        model = SMPL(smpl_dir, numImgsInSplit)
        # Freeze the rest of the SMPL model parameters
        for param in model.parameters():
            param.requires_grad = False  # Freeze all SMPL parameters

        model = model.to(device=device)  # move the model parameters to CPU/GPU


        for iEpoch in range(50):
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
                cam_intr = torch.from_numpy(dataset_train.cam_intr).float()
                H, W = 256, 256
                joints3d_trans = joints3d + interpolated_ori_trans[startImg:(startImg+numImgsInSplit)].unsqueeze(1).repeat(1, joints3d.shape[1], 1)
                joints2d = projection_torch(joints3d_trans, cam_intr, H, W)
                verts2d = projection_torch(verts, cam_intr, H, W)
                # verts2d_pix = (verts2d*H).type(torch.uint8).detach().cpu().numpy()
                verts2d_pix = (verts2d*H).detach().cpu().numpy() 

                # print('===== Ecor =====')
                # Initialize the scalar to store the cumulative loss
                loss_cor = torch.tensor(0.0, requires_grad=True)
                # params for ShiTomasi corner detection
                feature_params = dict( maxCorners = 100,
                                    qualityLevel = 0.3,
                                    minDistance = 7,
                                    blockSize = 7 )

                # Parameters for lucas kanade optical flow
                lk_params = dict( winSize  = (15, 15),
                                maxLevel = 2,
                                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                
                # Take first frame and find corners in it
                if os.path.exists('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg)):
                    old_gray = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                else:
                    print('the first image of sequence doesnt exist')
                p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
                
                # features on image using cv2 tracker
                featImgLocs.append(p0.reshape(-1,2))

                for iImg in range(1, numImgsInSplit):
                    if os.path.exists('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iImg)):
                        frame_gray = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+iImg), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                    else:
                        print('next image doesnt exist')
                    # calculate optical flow
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                    # Select good points
                    if p1 is not None:
                        good_new = p1[st==1]
                        good_old = p0[st==1]
                    # Now update the previous frame and previous points
                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)
                    featImgLocs.append(p0.reshape(-1,2))

                    tolerance = 10
                    if (0):
                        distances, closest_vert_indices = findCloestPoint(verts2d_pix[iImg], p0.squeeze(1))
                        selectedDistancesSqSum = np.sum(distances[distances < tolerance]**2)
                    p0tensor = torch.tensor(p0.squeeze(1), requires_grad=False).to(device)  # (n, 2)
                    min_distances, closest_vert_indices = findClosestPointTorch(verts2d[iImg]*H, p0tensor)
                    selectedDistancesSqSum = torch.sum(min_distances[min_distances < tolerance]**2)
                    loss_cor = loss_cor + selectedDistancesSqSum

                # print('===== Etemp =====')
                loss_temp = torch.norm((joints2d[1:] - joints2d[:-1]), dim=2).sum()


                lossFor1Seq = loss_cor + 10*loss_temp
                # lossFor1Seq = loss_cor
                colors = {feature_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for feature_id in range(len(p0))}
                min_distances = min_distances.cpu().detach().numpy()
                featAndVertOnImg = draw_feature_dots(frame_gray, p0.squeeze(1)[min_distances < tolerance], \
                                                     (verts2d[-1]*H).type(torch.uint8).detach().cpu().numpy(), \
                                                        closest_vert_indices[min_distances < tolerance], colors)
                jointsForVisual = joints2d[-1].detach().cpu().numpy() * 256
                jointsForVisual = jointsForVisual.astype(np.uint8)
                
                skeletonOnImg = draw_skeleton(frame_gray, jointsForVisual, draw_edges=True)

                writer.add_images('feat and closest vertics' , featAndVertOnImg, iEpoch*sequence_length + startImg, dataformats='HWC')
                writer.add_images('skeleton' , skeletonOnImg, iEpoch*sequence_length + startImg, dataformats='HWC')
                writer.add_scalar('training_loss', lossFor1Seq.item(), iEpoch*sequence_length + startImg)

                total_loss.append(lossFor1Seq.item())
                lossFor1Seq.backward()
                learnable_params[0].grad
                optimizer.step()
                optimizer.zero_grad()

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
        for iEpoch in range(20):
            total_loss = []

            for iSplit in range(totalSplits):
                print('===== Esil =====')
                loss_sil = torch.tensor(0.0, requires_grad=True)
                startImg = iSplit * numImgsInSplit
                interpolated_rotmats = batch_rodrigues(learnable_pose_and_shape[startImg:(startImg+numImgsInSplit), 3:75].reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
                interpolated_rotmats.shape
                verts, joints3d, _ = model(beta=learnable_pose_and_shape[startImg:(startImg+numImgsInSplit),75:85].view(-1, 10),
                                    theta=None,
                                    get_skin=True,
                                    rotmats=interpolated_rotmats.view(-1, 24, 3, 3))
                verts = verts + interpolated_ori_trans[startImg:(startImg+numImgsInSplit)].unsqueeze(1).repeat(1, verts.shape[1], 1)
                cam_intr = torch.from_numpy(dataset_train.cam_intr).float()
                H, W = 256, 256
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
                
                print('===== Estab =====')
                loss_stab = torch.sum(torch.norm(joints2d - init_joints2d[startImg:(startImg+numImgsInSplit)], dim=2)**2)
                joints2d.shape
                init_joints2d[startImg:(startImg+numImgsInSplit)].shape
                loss_refined = loss_sil + loss_stab
                # loss_refined = loss_sil
                loss_refined.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss.append(loss_refined.item())

                if os.path.exists('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+numImgsInSplit-1)):
                    frame_gray = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, startImg+numImgsInSplit), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
                colors = {feature_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for feature_id in range(closest_events_u.shape[0])}
                # boundaryAndEventOnImg = draw_feature_dots(frame_gray, np.transpose(boundaryPixelsinSeq[-1], (1, 0)), \
                #                                      closest_events_u, \
                #                                         [x for x in range(closest_events_u.shape[0])], colors)
                boundaryAndEventOnImg = draw_feature_dots(frame_gray, boundaryPixelsinSeq[-1][:,[1,0]], closest_events_u, [x for x in range(closest_events_u.shape[0])], colors)
                # boundaryAndEventOnImg = draw_feature_dots(frame_gray, boundaryPixelsinSeq[-1], closest_events_u, [x for x in range(closest_events_u.shape[0])], colors)
                
                closest_events_u[:,1].min()
                boundaryPixelsinSeq[-1].shape
                boundaryPixelsinSeq[-1][0]
                closest_events_u[0]
                # cv2.imwrite('tmp.jpg', boundaryAndEventOnImg)
                writer.add_images('boundary_on_Img' , boundaryAndEventOnImg, iEpoch*sequence_length + startImg, dataformats='HWC')
                writer.add_scalar('refinement_loss', loss_refined.item(), iEpoch*sequence_length + startImg)

            averLossForEpoch = sum(total_loss) / len(total_loss)
            print(iEpoch, " aver loss for Etemp & Ecor: ", averLossForEpoch)


def write_tensorboard(writer, results, epoch, progress, mode, args):
    action, sample_frames_idx = results['info']
    verts = results['verts'].cpu().numpy()  # [T+1, 6890, 3]
    cam_intr = results['cam_intr'].cpu().numpy()
    faces = results['faces']

    fullpics, render_imgs = [], []
    for i, frame_idx in enumerate(sample_frames_idx):
        img = cv2.imread('%s/full_pic_%i/%s/fullpic%04i.jpg' % (args.data_dir, args.img_size, action, frame_idx))
        fullpics.append(img[:, :, 0:1])

        vert = verts[i]
        dist = np.abs(np.mean(vert, axis=0)[2])
        # render_img = (util.render_model(vert, faces, args.img_size, args.img_size, cam_intr, np.zeros([3]),
        #                                 np.zeros([3]), near=0.1, far=20 + dist, img=img) * 255).astype(np.uint8)
        render_img = util.render_model(vert, faces, args.img_size, args.img_size, cam_intr, np.zeros([3]),
                                       np.zeros([3]), near=0.1, far=20 + dist, img=img)
        render_imgs.append(render_img)

    fullpics = np.transpose(np.stack(fullpics, axis=0), [0, 3, 1, 2]) / 255.
    fullpics = np.concatenate([fullpics, fullpics, fullpics], axis=1)
    writer.add_images('%s/fullpic%06i' % (mode, epoch * 100 + progress), fullpics, 1, dataformats='NCHW')

    render_imgs = np.transpose(np.stack(render_imgs, axis=0), [0, 3, 1, 2])
    writer.add_images('%s/shape%06i' % (mode, epoch * 100 + progress), render_imgs, 1, dataformats='NCHW')


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
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--smpl_dir', type=str, default='/home/ziyan/02_research/EventHPE/smpl_model/basicModel_m_lbs_10_207_0_v1.0.0.pkl')
    parser.add_argument('--num_worker', type=int, default=0)
    parser.add_argument('--pin_memory', type=int, default=1)
    parser.add_argument('--use_amp', type=int, default=1)

    parser.add_argument('--events_input_channel', type=int, default=8)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--max_steps', type=int, default=8)
    parser.add_argument('--num_steps', type=int, default=8)
    parser.add_argument('--skip', type=int, default=2)
    parser.add_argument('--use_hmr_feats', type=int, default=1)
    parser.add_argument('--use_flow', type=int, default=1)
    parser.add_argument('--use_flow_rgb', type=int, default=0)
    parser.add_argument('--use_geodesic_loss', type=int, default=1)
    parser.add_argument('--vibe_regressor', type=int, default=0)
    parser.add_argument('--use_vibe_init', type=int, default=0)
    parser.add_argument('--use_hmr_init', type=int, default=0)

    parser.add_argument('--delta_tran_loss', type=float, default=0)
    parser.add_argument('--tran_loss', type=float, default=1)
    parser.add_argument('--theta_loss', type=float, default=10)
    parser.add_argument('--joints3d_loss', type=float, default=1)
    parser.add_argument('--joints2d_loss', type=float, default=10)
    parser.add_argument('--flow_loss', type=float, default=0.1)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr_start', '-lr', type=float, default=0.01)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--lr_decay_step', type=float, default=1)
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
