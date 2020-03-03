# imports
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import glob

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import os
import numpy as np
import json
from detectron2.structures import BoxMode

classes = { 'Car': 0, 
            'Van': 1,
            'Truck': 2,
            'Pedestrian': 3,
            'Person_sitting': 4,
            'Cyclist': 5,
            'Tram': 6,
            'Misc': 7,
            'DontCare': 8 }

def get_KITTI_dicts(img_dir, is_train):

    # Use glob to obtain the paths
    image_paths = sorted(glob.glob(img_dir + '*'))
    label_paths = []
    for path in image_paths :
        splitd = path.split(os.sep)
        img_name = splitd[-1]
        img_id = img_name.split('.')[0]
        label_paths.append('/home/mcv/datasets/KITTI/training/label_2/' + img_id + '.txt')

    # Randomize the list
    random.seed(42)
    ids = list(range(0,len(label_paths)))
    random.shuffle(ids)

    if(is_train):
        ids = ids[0:round(0.8*len(ids))]
    else:
        ids = ids[(round(0.8*len(ids))+1):-1]

    #Create dict
    dataset_dicts = []

    # Iterate through the images
    for i in ids:
        record = {}
        height, width = cv2.imread(image_paths[i]).shape[:2]

        record["file_name"] = image_paths[i]
        record["image_id"] = i
        record["height"] = height
        record["width"] = width

        # Read the txt
        with open(label_paths[i]) as f:
            content = f.readlines()   
        content = [x.strip() for x in content] 

        objs = []
        for line in content:
            anno = line.split()

            obj = {
                "bbox": [anno[4], anno[5], anno[6], anno[7]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": classes[anno[0]]
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return(dataset_dicts)


from detectron2.data import DatasetCatalog, MetadataCatalog


class_list =   ['Car', 
                'Van',
                'Truck',
                'Pedestrian',
                'Person_sitting',
                'Cyclist',
                'Tram',
                'Misc',
                'DontCare']

d = "train"
DatasetCatalog.register("KITTI_" + d, 
    lambda d=d: get_KITTI_dicts("/home/mcv/datasets/KITTI/data_object_image_2/mini_train/", True))
MetadataCatalog.get("KITTI_" + d).set(thing_classes=class_list)

d = "val"
DatasetCatalog.register("KITTI_" + d, 
    lambda d=d: get_KITTI_dicts("/home/mcv/datasets/KITTI/data_object_image_2/mini_train/", False))
MetadataCatalog.get("KITTI_" + d).set(thing_classes=class_list)

KITTI_metadata = MetadataCatalog.get("KITTI_train")