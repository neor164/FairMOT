from ast import Str
from copy import deepcopy
import itertools
import os
import os.path as osp
from re import T
import time
from collections import deque
from typing import Dict, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from models import *
from models.decode import mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.utils import *
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from matplotlib.patches import Rectangle
from tracker import matching
import tracker.tracker_data as td 
from .basetrack import BaseTrack, TrackState
import pandas as pd

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30, key_points:Optional[np.ndarray]=None, det_id:int=0):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.det_id = det_id
        self.score = score
        self.tracklet_len = 0
        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9
        self.key_points = key_points
        self.prev_bb = tlwh


    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks, kalman_data=None):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
                if kalman_data and stracks[i].track_id == kalman_data['tracker_id']:
                    kalman_data['predict'][stracks[i].frame_id] = stracks[i].tlwh
                diff = stracks[i].tlwh[:2] - stracks[i].prev_bb[:2]
                stracks[i].prev_bb = stracks[i].tlwh
                if  stracks[i].key_points is not None:
                    stracks[i].key_points[:, :2] += diff

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=True, update_weight=1.0,kalman_data=None,update_type='fairmot'):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
       
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh),update_weight)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.key_points = new_track.key_points
        if kalman_data is not None and self.track_id == kalman_data['tracker_id']:
            kalman_data['update'][frame_id] = self.tlwh
            kalman_data['covariance_error'][frame_id] = self.covariance[0,0]
            kalman_data['detection'][frame_id] = new_track.tlwh[0]
            # kalman_data['value'][frame_id] =  self.tlwh_to_xyah(new_tlwh)
            kalman_data['update_type'][frame_id] = update_type
        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)
        self.prev_bb = self.tlwh


    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.kalman_data: Dict[str,Dict[int, Union[np.ndarray, str]]] = {
        'predict':{},
        'update':{},
        'detection':{},
        'update_type':{},
        'gt':{},
        'tracker_id':None,
        'covariance_error':{}

        }
        self.scenario_data = {}
        self.matching_data = {}
        self.matched_target_id = False
        self.start_frame_target_id = opt.gt.loc[opt.gt['target_id'] == opt.tracker_id]['frame_id'].min()
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.fois_detector = opt.fois_detector
        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        
        self.kalman_filter = KalmanFilter()

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def update(self, im_blob, img0):
        frame = deepcopy(img0)
        self.frame_id += 1
        self.scenario_data[self.frame_id+1] = {'detections':[], 'ground_truth':{}, 'trackers':{}, 'keypoints':[]}

        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        text_scale = max(1, frame.shape[1] / 1600.)
        text_thickness = 2
        frame =  cv2.putText(frame, str(self.frame_id), (10, 10), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                                    thickness=text_thickness) 
        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)
            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        # vis
        '''
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img0, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
        cv2.imshow('dets', img0)
        cv2.waitKey(0)
        id0 = id0-1
        '''

        if len(dets) > 0:
            '''Detections'''
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30,det_id=idx) for
                          (idx,(tlbrs, f)) in enumerate(zip(dets[:, :5], id_feature))]
            
        else:
            detections = []

        self.scenario_data[self.frame_id+1]['detections']  = [td.DetectionsData(bounding_box=f.tlwh.tolist(), confidance=f.score).dict() for f in detections] 
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        if self.opt.use_kp:
            bbs, vkps = self.fois_detector[self.frame_id]
            for v in vkps:
                self.scenario_data[self.frame_id+1]['keypoints'].append(td.KeypointsData(keypoints=v.tolist()).dict())
            if bbs.size:
                track_bbs = [STrack(bb[:4], 1, 2, 30) for bb in bbs]
                dists_kp = matching.iou_distance(track_bbs, detections)
                matches, u_track, _ = matching.linear_assignment(dists_kp, thresh=0.9)
                for itracked, idet in matches:
                   detections[idet].key_points =  vkps[itracked]
                   self.scenario_data[self.frame_id+1]['keypoints'][itracked]['matched_with_detection']= True
                   self.scenario_data[self.frame_id+1]['detections'][idet]['keypoints'] = detections[idet].key_points.tolist()
                unmatches_keypoints = [vkps[ut] for ut in  u_track]   
                


        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        #for strack in strack_pool:
            #strack.predict()
        STrack.multi_predict(strack_pool, self.kalman_data)
        dists = matching.embedding_distance(strack_pool, detections)
        #dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.4)



        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id,kalman_data=self.kalman_data,update_type='fairmot')
                activated_starcks.append(track)
                bbox = track.tlwh
                self.scenario_data[self.frame_id+1]['detections'][det.det_id]['matched_with_tracker'] = True
                frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2] + bbox[0]), int(bbox[3]+bbox[1])),
                    (255, 0, 0), 2)
                text_scale = max(1, frame.shape[1] / 1600.)
                text_thickness = 2
                tid = str(int(track.track_id))
                frame =  cv2.putText(frame, tid, (int(bbox[0]-20), int(bbox[1] + 30)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                                    thickness=text_thickness)  
                # frame =  cv2.putText(frame, iou_text, (int(bbox[0]), int(bbox[1] + 30)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                #             thickness=text_thickness)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)




        # if len(r_tracked_stracks):
        #     fig, ax = plt.subplots(1,1,figsize=(20,12))
        #     fig.tight_layout()
        #     ax.imshow(img0)
        #     ax.axis('off')
        #     for bb in bbs:
        #         if bb[2]*bb[3] > 15:
        #             rectangle = Rectangle((bb[0], bb[1]), bb[2], bb[3],fill=False, edgecolor=(1,0,0))
        #             ax.add_patch(rectangle)
        #     for track in r_tracked_stracks:
        #         bb = track.tlwh
        #         rectangle = Rectangle((bb[0], bb[1]), bb[2], bb[3],fill=False, edgecolor=(0,0,1))
        #         ax.add_patch(rectangle)

        #     plt.savefig('temp.png')
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
       

        ''' Step 3: Second association, with IOU'''
        # r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        detections = [detections[i] for i in u_detection]

        dists = matching.iou_distance(r_tracked_stracks, detections)

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        # matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id,kalman_data=self.kalman_data,update_type='IoU')
                activated_starcks.append(track)
                bbox = track.tlwh
                frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2] + bbox[0]), int(bbox[3]+bbox[1])),
                    (255, 0, 0), 2)
                text_scale = max(1, frame.shape[1] / 1600.)
                text_thickness = 2
                tid = str(int(track.track_id))
                frame =  cv2.putText(frame, tid, (int(bbox[0]-20), int(bbox[1] + 30)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                                    thickness=text_thickness)  
                # frame =  cv2.putText(frame, iou_text, (int(bbox[0]
                self.scenario_data[self.frame_id+1]['detections'][det.det_id]['matched_with_tracker'] = True
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)



        
        if self.opt.use_kp:
           r_tracked_stracks = [r_tracked_stracks[i] for i in u_track if r_tracked_stracks[i].state == TrackState.Tracked]
           N = len(unmatches_keypoints)
           M = len(r_tracked_stracks)
           if N >0 and M>0:
            euc_dist = 4*np.ones([N,M])
            for n in range(N):
                for m in range(M):
                    ukps = unmatches_keypoints[n][:,:2]
                    track = r_tracked_stracks[m]
                    if track.key_points is not None:
                        rkps = track.key_points[:,:2]
                        diff = ukps - rkps
                        dist = np.sqrt(np.sum(( diff )**2,axis=1))
                        vis = np.argwhere( (unmatches_keypoints[n][:,2] ==2) & (track.key_points[:,2] ==2))
                        if len(vis) > 0:

                            diag = np.sqrt(track.tlwh[2]**2 + track.tlwh[3]**2) * (self.frame_id - track.frame_id)
                            edist = np.sum(dist[vis])/(len(vis)*diag)
                            euc_dist[n,m] = edist

            matches, u_keypoints, u_track = matching.linear_assignment(euc_dist, thresh=0.1)
            for idet, itracked in matches:
                
                track = r_tracked_stracks[itracked]

                ukps = unmatches_keypoints[idet][:,:2]
                rkps = r_tracked_stracks[itracked].key_points[:,:2]
                diff = ukps - rkps
                unvis = np.argwhere( (unmatches_keypoints[idet][:,2] ==2) | (r_tracked_stracks[itracked].key_points[:,2] ==2))
                n_vis = len(unvis)
                mdiff = np.mean(diff[unvis],axis=0)
                new_bb = deepcopy(track.prev_bb)
                new_bb[:2] += mdiff.flatten()
                new_track = STrack(new_bb, n_vis * self.opt.nkps_to_confidance, 2, 30,key_points=unmatches_keypoints[idet])
                track.update(new_track, self.frame_id,update_feature=False,kalman_data=self.kalman_data,update_type='keypoints',update_weight=new_track.score)
                bbox = track.tlwh
                frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2] + bbox[0]), int(bbox[3]+bbox[1])),
                    (0, 255, 0), 2)
                text_scale = max(1, frame.shape[1] / 1600.)
                text_thickness = 2
                euc_dist_text = str(round(euc_dist[ idet,itracked],2))
                frame =  cv2.putText(frame, euc_dist_text, (int(bbox[0]), int(bbox[1] + 30)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                            thickness=text_thickness)
                tid = str(int(track.track_id))
                frame =  cv2.putText(frame, tid, (int(bbox[0]-20), int(bbox[1] + 30)), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 255, 255),
                            thickness=text_thickness) 
                self.fois_detector.draw_keypoints(frame, unmatches_keypoints[idet],color=[0,255,0])
                activated_starcks.append(track)

            for  idet in u_keypoints:

                self.fois_detector.draw_keypoints(frame, unmatches_keypoints[idet])
       

                        
        for it in u_track:
            track = r_tracked_stracks[it]
            # bbox = track.tlwh

            # frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
            #             (int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])),
            #             (0, 255, 255), 2) 
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)


        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
            
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        
        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        ## match with gt
        # gt_row  = self.opt.gt.loc[(self.opt.gt[0]== self.frame_id) & (self.opt.gt[1]== self.opt.tracker_id)]
        # if len(gt_row):
        #     gt_bb = [int(gt_row[2]), int(gt_row[3]),int(gt_row[4]), int(gt_row[5])]
        #     gt_t = STrack(gt_bb, 1, 1, 30)
        # if not self.matched_target_id and self.start_frame_target_id <= self.frame_id and  len(gt_row):

        #         dists = matching.iou_distance(activated_starcks,[gt_t])
        #         matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.2)
        #         for itracked, idet in matches:
        #             self.matched_target_id = True
        #             self.kalman_data['tracker_id'] = activated_starcks[itracked].track_id


        # if self.matched_target_id and len(gt_row):
        #     self.kalman_data['gt'][self.frame_id] = gt_bb[0]
            # frame = cv2.rectangle(frame, (int(gt_bb[0]), int(gt_bb[1])),
            #             (int(gt_bb[2]+gt_bb[0]), int(gt_bb[3]+gt_bb[1])),
            #             (0, 255, 0), 2)
            # for tracker in activated_starcks:
            #     if self.kalman_data['tracker_id'] == tracker.track_id:
            #         bbox = tracker.tlwh
            #         # bbox = track.tlwh
            #         if tracker.end_frame < self.frame_id:
            #             color = [0,255,255]
            #         else:
            #             color = [0,0,255]
            #         # frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
            #         #             (int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])),
            #         #             color, 2) 
           


        """ Step 5: Update state"""
        pred_tracks = []
        for track in self.lost_stracks:
            bbox = track.tlwh


            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
                frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])),
                        (0, 100, 255), 2) 
                if track.key_points is not None:
                    self.fois_detector.draw_keypoints(frame, track.key_points, color= [0,100,255])
                
            else:
                frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])),
                        (0, 255, 255), 2) 
                if track.key_points is not None:
                    self.fois_detector.draw_keypoints(frame, track.key_points, color= [0,255,255])
                pred_tracks.append(track)
        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        eval_tracks = deepcopy(output_stracks)
        eval_tracks.extend(pred_tracks)
        for track in eval_tracks:
            self.scenario_data[self.frame_id + 1]['trackers'][track.track_id] = td.TrackerData(id=track.track_id,bounding_box=track.tlwh.tolist(), updated=True if track.frame_id == self.frame_id else False).dict()
            if isinstance(track.key_points, np.ndarray):
                self.scenario_data[self.frame_id+ 1]['trackers'][track.track_id]['keypoints'] = track.key_points.tolist()

        gt_df:pd.Dataframe  = self.opt.gt.loc[(self.opt.gt['frame_id']== self.frame_id + 1)& (self.opt.gt['vis']>0)]
        gts = []
        for idx, gt_row in gt_df.iterrows():

            if len(gt_row):
                gt_bb = [int(gt_row[2]), int(gt_row[3]),int(gt_row[4]), int(gt_row[5])]
                gts.append(STrack(gt_bb, 1, 1, 30,det_id=gt_row[1]))
                self.scenario_data[self.frame_id+ 1]['ground_truth'][int(gt_row[1])] = td.GroundTruthData(id=int(gt_row[1]), bounding_box=gt_bb).dict() 

        dists = matching.iou_distance(eval_tracks, gts)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track:STrack = eval_tracks[itracked]
            gtid = gt_df.iloc[idet]['target_id']
            self.scenario_data[self.frame_id+1]['trackers'][track.track_id]['matched_with_gt'] = int(gtid)
            self.scenario_data[self.frame_id+1]['ground_truth'][gtid]['matched_with_tracker'] = int(track.track_id)
            if track.tracklet_len == 0:
                if not track.track_id in self.matching_data:
                    self.matching_data[track.track_id] = {}
                self.matching_data[track.track_id]['gt'] = int(gtid)
            idx = self.opt.gt.loc[(self.opt.gt['frame_id']==self.frame_id + 1) & (self.opt.gt['target_id'] == gtid )].index    
            self.opt.gt.at[idx, 'matched_tracker'] = int(track.track_id)
    
        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format([track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))
        cv2.imwrite(os.path.join(self.opt.kp_data, '{:05d}.jpg'.format(self.frame_id)), frame)
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
