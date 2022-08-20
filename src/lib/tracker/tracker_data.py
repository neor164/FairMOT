from pydantic import BaseModel
from typing import Optional, List

class TrackerData(BaseModel):
    id: int
    bounding_box:Optional[List[int]] = None
    updated:bool=False
    keypoints:Optional[List[List[float]]] = None
    matched_with_gt:bool = False

class DetectionsData(BaseModel):
    
    bounding_box:List[int] = None
    keypoints:Optional[List[List[float]]] = None
    matched_with_tracker:bool = False
    confidance:float

class KeypointsData(BaseModel):
    keypoints:Optional[List[List[float]]] = None
    matched_with_detection:bool = False

class GroundTruthData(BaseModel):
    id:int
    bounding_box:List[int]
    matched_with_tracker:bool = False