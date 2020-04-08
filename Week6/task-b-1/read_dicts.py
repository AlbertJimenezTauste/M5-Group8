# imports
import torch
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
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.evaluation import DatasetEvaluators, DatasetEvaluator

# utils to read
import load_things as io
import pycocotools.mask as rletools
import os
import numpy as np
import json
from detectron2.structures import BoxMode
import PIL.Image as Image

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import ColorMode

def get_vKITTI_dicts(dataset_dir):
    # Obtain all the data from the seqmap
    seqmap = ['Scene01','Scene02','Scene06','Scene18','Scene20']

    # Iterate through all the sequences
    dataset_dicts = [] 
    for seq in seqmap: 
        print(seq)
        img_list = sorted(glob.glob(os.path.join(dataset_dir, seq, 'clone/frames/rgb/Camera_0/*.jpg')))
        gt_img_list = sorted(glob.glob(os.path.join(dataset_dir, seq, 'clone/frames/instanceSegmentation/Camera_0/*.png')))
        print(os.path.join(dataset_dir, seq, 'clone/frames/rgb/Camera_0/*.jpg'))
        print(os.path.join(dataset_dir, seq, 'clone/frames/instanceSegmentation/Camera_0/*.png'))
        print(len(gt_img_list))
        print(len(img_list))

        for i, gt_img_path in enumerate(gt_img_list):
            record = {}
            img = np.array(Image.open(gt_img_path))
            height, width = img.shape[:2]
            record["file_name"] = img_list[i]
            record["image_id"] = seq + str(i)
            record["height"] = height
            record["width"] = width

            instances = np.unique(img)
            objs = []
            obj_num = 0
            mask = np.zeros(img.shape, dtype=np.uint8, order="F")
            for idx, obj_id in enumerate(instances):
                if obj_id == 0:  # background
                    continue
                mask.fill(0)
                pixels_of_elem = np.where(img == obj_id)
                mask[pixels_of_elem] = 1

                # Create segmentation data
                contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                segmentation = []
                for contour in contours:
                    if contour.size >= 6:
                        segmentation.append(contour.flatten().tolist())
                if segmentation == []: continue

                RLEs = rletools.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
                RLE = rletools.merge(RLEs)
                area = rletools.area(RLE)
                [x, y, w, h] = cv2.boundingRect(mask)

                obj = {
                    "segmentation": segmentation,  # poly
                    "area": area,  # segmentation area
                    "iscrowd" : 0,
                    "bbox": [np.min(pixels_of_elem[1]), np.min(pixels_of_elem[0]), np.max(pixels_of_elem[1]), np.max(pixels_of_elem[0])],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 1, #obj_id // 1000,
                    "id": obj_num
                }
                obj_num = obj_num+1
                objs.append(obj)

            record["annotations"] = objs
            dataset_dicts.append(record)

    print('Loaded ' + str(np.shape(dataset_dicts)[0]) + ' images!')
    return(dataset_dicts)

def get_KITTI_MOTS_dicts(img_dir, seqmap):
    MODE = 2

    # Obtain all the data from the seqmap
    path = '/home/grupo08/datasets/KITTI_MOTS/instances_txt/'
    objects = io.load_sequences(path, seqmap)

    # Iterate through all the sequences
    dataset_dicts = []
    for seq in seqmap:
        print(seq)
        random.seed(42)
        ids = list(objects[seq].keys())
        random.shuffle(ids)

        # Iterate through the frames in ids (train or val)
        for i in ids:
            record = {}
            #height, width = cv2.imread(image_paths[i]).shape[:2]
            record["file_name"] = img_dir + seq + '/' + f'{i:06}' +'.png'
            record["image_id"] = seq + str(i)
            record["height"] = objects[seq][i][0].mask['size'][0]
            record["width"] = objects[seq][i][0].mask['size'][1]
            #record["sem_seg_file_name"] = img_dir + seq + '/mask_' + f'{i:06}' +'.png'

            # Iterate through all the instances in the i-th frame for sequence seq
            objs = []
            obj_num = 0
            for instance in objects[seq][i]:
                # class_id=10 means ignore instance
                if(instance.class_id==10):
                    continue
                # Decode the mask
                mask = rletools.decode(instance.mask)
                mask = np.array(mask)
                # Find the edges of the mask
                pos = np.where(mask>0)
                tl = [np.min(pos[0]), np.min(pos[1])]
                br = [np.max(pos[0]), np.max(pos[1])]

                if MODE == 0:
                    obj = {
                        "bbox": [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": instance.class_id
                    }

                elif MODE == 1:
                    obj = {
                        "bbox": [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": instance.class_id
                    }
                elif MODE == 2:
                    # Create segmentation data
                    contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                   
                    segmentation = []

                    for contour in contours:
                        if contour.size >= 6:
                            segmentation.append(contour.flatten().tolist())

                    if segmentation == []: continue

                    RLEs = rletools.frPyObjects(segmentation, mask.shape[0], mask.shape[1])
                    RLE = rletools.merge(RLEs)
                    # RLE = rletools.encode(np.asfortranarray(mask))
                    area = rletools.area(RLE)
                    [x, y, w, h] = cv2.boundingRect(mask)

                    '''
                    if mask is not None:
                        p_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
                        cv2.drawContours(p_mask, contours, -1, (0,255,0), 1)
                        cv2.rectangle(p_mask,(x,y),(x+w,y+h), (255,0,0), 2)
                        cv2.imwrite("../" + short_model_name + "/masks/mask_" + str(i) + ".jpg", p_mask)
                    '''
                    obj = {
                        "segmentation": segmentation,  # poly
                        "area": area,  # segmentation area
                        "iscrowd" : 0,
                        "bbox": [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": instance.class_id,
                        "id": obj_num
                    }

                    obj_num = obj_num+1

                elif MODE == 3:
                    obj = {
                        "bbox": [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": instance.class_id,
                        "segmentation": rletools.encode(np.asarray(mask, order="F"))
                    }

                elif MODE == 4:
                    obj = {
                        "bbox": [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": instance.class_id,
                        "iscrowd": 0,
                        "segmentation": instance.mask
                    }
    
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    
    print('Loaded ' + str(np.shape(dataset_dicts)[0]) + ' images!')
    return(dataset_dicts)