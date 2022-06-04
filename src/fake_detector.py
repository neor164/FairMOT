import json
from typing import Tuple, List
import numpy as np
import random

def get_boundingbox_kp(kp:np.ndarray)->List[int]:
    vis_kp = kp[kp[:,2] == 2]
    if vis_kp.size:
        min_x,min_y, _ = vis_kp.min(axis=0)
        max_x,max_y, _ = vis_kp.max(axis=0)

    # np.mean(kp[kp[:,2] == 1],axis=0)

        return [min_x, min_y, max_x - min_x, max_y - min_y]

class MotSynthDetector:
    def __init__(self,ann_path:str, alpha:float=1.75) -> None:
        self.alpha = alpha
        self.ann_path = ann_path
        with open(self.ann_path, 'r' ) as w:
            data = json.loads(w.read())
            self.annotations = np.array(data['annotations'])
            self.video_info = data['info']
            self.images_info = np.array(data['images'])
            self.additional_data = np.array(data['categories'])
        self.ann_frame_ids = np.array([f['image_id']  for  f in self.annotations],dtype=np.int32)
        self.frame_ids = np.unique(self.ann_frame_ids)
        self.min_frame = self.frame_ids.min()


    def get_bbs_by_frame(self, frame_id:int)-> np.ndarray:
        inds = np.argwhere(self.ann_frame_ids - self.min_frame +1 == frame_id)
        # if len(inds):
        #     # prob_vec = np.random.rand(len(inds))
        #     # inds = inds[prob_vec <= self.precision]
        #     out_inds = []
        #     for ind in inds:
        #         kp = np.array(self.annotations[ind][0]['keypoints']).reshape(-1, 3)
        #         likelihood = self.alpha * np.count_nonzero(kp[:, 2] == 2) / kp[:, 2].size
        #         if random.random() <= likelihood:
        #             out_inds.extend(ind)
        anns = self.annotations[inds]
        bbs = np.empty((0, 4))

        for ann in anns:
            bb = get_boundingbox_kp(np.array(ann[0]['keypoints']).reshape(22,3))
            if bb:
                bb = np.array([bb])
                bbs = np.concatenate((bbs,bb),axis=0)

        return bbs
        

    def get_frame_path(self, frame_id:int):
        ind = frame_id - self.min_frame 
        return self.images_info[ind]['file_name']

    @property
    def resolution(self) -> Tuple[int, int]:
        return ( self.video_info['img_width'], self.video_info['img_height'])
    
    def __getitem__(self, frame_id:int)->np.ndarray:
        return self.get_bbs_by_frame(frame_id)

