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

# utils to read
import load_things as io
import pycocotools.mask as rletools

import os
import numpy as np
import json
from detectron2.structures import BoxMode

"""
classes = { 'Pedestrian': 2, 
            'Car': 1 }
"""


# Mode selection
# 0 detection with class mapping
# 1 detection without class mapping
# 2 segmentation
MODE = 2

def get_KITTI_MOTS_dicts(img_dir, seqmap):

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

                print(mask.shape)

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


from detectron2.data import DatasetCatalog, MetadataCatalog

# 0-->None, 1-->Car, 2--->Pedestrian
class_list =   ['None','Car','Pedestrian']
# Sequences to load:
train_seqmap = ['0000','0001','0003','0004','0005','0009','0011','0012','0015','0017','0019','0020']

val_seqmap = ['0002','0006','0007','0008','0010','0013','0014','0016','0018']


# Create the dataset
d = "train"
DatasetCatalog.register("KITTI_MOTS_" + d, 
    lambda d=d: get_KITTI_MOTS_dicts("/home/mcv/datasets/KITTI-MOTS/training/image_02/", train_seqmap))
MetadataCatalog.get("KITTI_MOTS_" + d).set(thing_classes=class_list)

d = "val"
DatasetCatalog.register("KITTI_MOTS_" + d, 
    lambda d=d: get_KITTI_MOTS_dicts("/home/mcv/datasets/KITTI-MOTS/training/image_02/", val_seqmap))
MetadataCatalog.get("KITTI_MOTS_" + d).set(thing_classes=class_list)

KITTI_metadata = MetadataCatalog.get("KITTI_MOTS_train")
print(KITTI_metadata)

# Verify it is well loaded
dataset_dicts = get_KITTI_MOTS_dicts("/home/mcv/datasets/KITTI-MOTS/training/image_02/", train_seqmap)
ide = 0
for d in random.sample(dataset_dicts, 2):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=KITTI_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite("img_"+str(ide)+".jpg", vis.get_image()[:, :, ::-1])
    ide += 1

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


cfg = get_cfg()

model_name = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
short_model_name = model_name.split("/")
short_model_name = short_model_name[1]
short_model_name = short_model_name[:-5]

os.makedirs("../gt_" + short_model_name, exist_ok=True)
os.makedirs("../mask_" + short_model_name, exist_ok=True)
os.makedirs("../inference_" + short_model_name, exist_ok=True)

cfg.merge_from_file(model_zoo.get_config_file(model_name))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)  # Let training initialize from model zoo
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model

cfg.DATASETS.TRAIN = ("KITTI_MOTS_train",)
cfg.DATASETS.TEST = ("KITTI_MOTS_val",)
cfg.DATALOADER.NUM_WORKERS = 2

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025 
cfg.SOLVER.MAX_ITER = 5000    
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  

cfg.INPUT.MASK_FORMAT='bitmask'


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.6 # to be effective in RetinaNet
cfg.DATASETS.TEST = ("KITTI_MOTS_val", )
predictor = DefaultPredictor(cfg)


from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_KITTI_MOTS_dicts("/home/mcv/datasets/KITTI-MOTS/training/image_02/", val_seqmap)
ide = 0
for d in random.sample(dataset_dicts, 20):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                    metadata=KITTI_metadata, 
                    scale=0.8
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("predicted/predicted"+str(ide)+".jpg", v.get_image()[:, :, ::-1])
    ide += 1


evaluator = COCOEvaluator("KITTI_MOTS_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "KITTI_MOTS_val")
inference_on_dataset(predictor.model, val_loader, evaluator)