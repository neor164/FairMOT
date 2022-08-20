from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cProfile import label
import json
from typing import Dict, Union

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
import pandas as pd
from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets
from fake_detector import MotSynthDetector
from tracking_utils.utils import mkdir_if_missing
from opts import opts


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30, use_cuda=True):
    if save_dir:
        mkdir_if_missing(save_dir)

    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    

    print(f'III) found GPU: {torch.cuda.is_available()}')
    
    #for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        #if i % 8 != 0:
            #continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        # bbs = fois_detector.get_bbs_by_frame(frame_id)
        # run tracking
        timer.tic()
        if use_cuda:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
        else:
            blob = torch.from_numpy(img).unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        #online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                #online_scores.append(t.score)
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    #write_results_score(result_filename, results, data_type)
    # frame_ids = list(tracker.kalman_data['predict'].keys())
    # min_frame = np.min(frame_ids)
    # max_frame = np.max(frame_ids)
    # pred_x = np.full([len(frame_ids)], np.nan)
    # update_x =  np.full([len(frame_ids)], np.nan)
    # detection_x = np.full([len(frame_ids)], np.nan)
    # has_detection = np.full([len(frame_ids)], np.nan)
    # no_detection = np.full([len(frame_ids)], np.nan)
    # iou_x = np.full([len(frame_ids)], np.nan)
    # keypoints_x = np.full([len(frame_ids)], np.nan)
    # covariance_x = np.full([len(frame_ids)], np.nan)

    # gt_x = np.full([len(frame_ids)], np.nan)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1,2, figsize=(28,14))
    # # fig.tight_layout()
    # for idx, fid in enumerate(frame_ids):
    #     pred_x[idx] = tracker.kalman_data['predict'][fid][0]
    # min_frame = np.min(list(tracker.kalman_data['update'].keys()))    
    # for key, value in tracker.kalman_data['update'].items():
        
    #     idx =  key - min_frame
    #     update_type =  tracker.kalman_data['update_type'].get(key, None)
    #     # if update_type.lower() == 'fairmot':
    #     #     fairmot_x[idx] = tracker.kalman_data['update'][key][0]
    #     # elif update_type.lower() == 'iou':
    #     #     iou_x[idx] = tracker.kalman_data['update'][key][0]
    #     # else:  
    #     if len(detection_x) > idx:       
    #         detection_x[idx] = tracker.kalman_data['detection'].get(key, np.nan)
    #     if len(update_x) > idx: 
    #         update_x[idx] = tracker.kalman_data['update'].get(key, np.nan)[0]
    #         covariance_x[idx] = tracker.kalman_data['covariance_error'].get(key, np.nan)
    # min_frame = np.min(list(tracker.kalman_data['gt'].keys()))     
    # for key, value in tracker.kalman_data['gt'].items():
    #     idx =  key - min_frame
    #     if len(gt_x) > idx: 
    #         gt_x[idx] = tracker.kalman_data['gt'].get(key, np.nan)

    data = {'frame_data':tracker.scenario_data,'matching_data':tracker.matching_data}

    return frame_id, timer.average_time, timer.calls, data


def main(opt, data_root='/data', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    print(f'III) found GPU: {torch.cuda.is_available()}')

    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        annotations_path = os.path.join(data_root, seq)
        opt.kp_data = '/data/kp_data'

        opt.gt = pd.read_csv('/data/MOTSynth/033/gt/gt.txt',names=['frame_id', 'target_id', 'x', 'y', 'w', 'h', 'y_pred', 'x_pred', 'vis', 'w_pred'],header=None, index_col=False)
        opt.gt['matched_tracker'] = None
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        if opt.mot_synth:
            data_path = osp.join(data_root,'imgs','frames', seq, 'rgb')
            # annotations_path = os.path.join(data_root,'mot_annotations', seq)
            anns_json_path = f'/app/FairMOT/annotations/{seq}.json'
            fois_detector = MotSynthDetector(anns_json_path, opt)
            opt.fois_detector = fois_detector
        else:
            data_path = osp.join(data_root, seq, 'img1')
        dataloader = datasets.LoadImages(data_path, opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        
        meta_info = open(os.path.join(annotations_path, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc,data = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        # data_root = '/data/MOTSynth'
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
          
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            kpout_path = osp.join(opt.kp_data, '{}.mp4'.format(seq + 'kp'))

            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            # os.system(cmd_str)

            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(opt.kp_data, kpout_path)
            os.system(cmd_str)

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))
    res_name = f'{seq}_results' if opt.use_kp else  f'{seq}_results_wo_kps'
    opt.gt.to_csv(res_name + '.csv')
    with open(res_name +'.json', 'w') as jf:
            json.dump(data, jf)

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    print(f'I) found GPU: {torch.cuda.is_available()}')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    opt = opts().init()
    print(f'II) found GPU: {torch.cuda.is_available()}')

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        #seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, 'MOT15/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12a
                      MOT16-14'''
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')


    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07y
                      MOT20-08
                      '''
    if opt.mot_synth:
        seqs_str = '''
                        033
                      '''
        data_root = os.path.join(opt.data_dir, 'MOTSynth')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOTSynth',
         show_image=False,
         save_images=False,
         save_videos=False)
    os.system('rm -r /data/results/*')
    # os.system('mv data/results/MOTSynth/* /data/results/')
    # os.system('mv /data/outputs/MOTSynth/*/*.mp4 /data/results/')
    os.system('mv /data/kp_data/033kp.mp4 /data/results/')
    os.system('rm -r /data/outputs/MOTSynth/* data/results/MOTSynth/')
    os.system('cd /data/results/ && zip -r /data/results.zip  /data/results/')

