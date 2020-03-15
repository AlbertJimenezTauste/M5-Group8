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
def get_MOTS_dicts(img_dir, seqmap):

    # Obtain all the data from the seqmap
    path = '/home/grupo08/datasets/MOTSChallenge/instances_txt/'
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
            record["file_name"] = img_dir + seq + '/' + f'{i:06}' +'.jpg'
            record["image_id"] = seq + str(i)
            record["height"] = objects[seq][i][0].mask['size'][0]
            record["width"] = objects[seq][i][0].mask['size'][1]

            # Iterate through all the instances in the i-th frame for sequence seq
            objs = []
            for instance in objects[seq][i]:
                # class_id=10 means ignore instance
                if(instance.class_id==10):
                    continue
                # Decode the mask
                x = rletools.decode(instance.mask)
                x = np.array(x)
                # Find the edges of the mask
                pos = np.where(x>0)
                tl = [np.min(pos[0]), np.min(pos[1])]
                br = [np.max(pos[0]), np.max(pos[1])]

                obj = {
                    "bbox": [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": instance.class_id
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
#seqmap = ['0002','0005','0009','0011']
train_seqmap = ['0002', '0009','0011']

val_seqmap = ['0005']


# Create the dataset
d = "train"
DatasetCatalog.register("MOTSChallenge_" + d, 
    lambda d=d: get_MOTS_dicts("/home/mcv/datasets/MOTSChallenge/train/images/", train_seqmap))
MetadataCatalog.get("MOTSChallenge_" + d).set(thing_classes=class_list)

d = "val"
DatasetCatalog.register("MOTSChallenge_" + d, 
    lambda d=d: get_MOTS_dicts("/home/mcv/datasets/MOTSChallenge/train/images/", val_seqmap))
MetadataCatalog.get("MOTSChallenge_" + d).set(thing_classes=class_list)

MOTS_metadata = MetadataCatalog.get("MOTSChallenge_train")
#MOTS_metadata_val = MetadataCatalog.get("MOTSChallenge_val")
print(MOTS_metadata)
#print(MOTS_metadata_val)


# Verify it is well loaded
dataset_dicts = get_MOTS_dicts("/home/mcv/datasets/MOTSChallenge/train/images/", train_seqmap)
#dataset_dicts_val = get_MOTS_dicts("/home/mcv/datasets/MOTSChallenge/train/images/", seqmap, is_train=False)


ide = 0
for d in random.sample(dataset_dicts, 10):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=MOTS_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite("examples_MOTS/img_"+ str(ide) + ".jpg", vis.get_image()[:, :, ::-1])
    ide += 1

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
#cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("MOTSChallenge_train",)
cfg.DATASETS.TEST = ("MOTSChallenge_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (ballon)


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model
cfg.DATASETS.TEST = ("MOTSChallenge_val", )
predictor = DefaultPredictor(cfg)


from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_MOTS_dicts("/home/mcv/datasets/MOTSChallenge/train/images/", val_seqmap)
ide = 0
for d in random.sample(dataset_dicts, 20):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                    metadata=MOTS_metadata, 
                    scale=0.8
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("predicted_MOTS/predicted"+str(ide)+".jpg", v.get_image()[:, :, ::-1])
    ide += 1

evaluator = COCOEvaluator("MOTSChallenge_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "MOTSChallenge_val")
inference_on_dataset(predictor.model, val_loader, evaluator)
