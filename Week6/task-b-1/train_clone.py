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

# Custom trainer class
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR,'validation')
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)


# 0-->None, 1-->Car, 2--->Pedestrian
class_list =   ['None','Car','Pedestrian']
# Sequences to load:
train_seqmap = ['0001','0002','0006','0018','0020']
val_seqmap = ['0000','0003','0010','0012','0014']
test_seqmap = ['0004','0005', '0007', '0008', '0009', '0011', '0015']

# Base model name
model_name = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
short_model_name = model_name.split("/")
short_model_name = short_model_name[1]
short_model_name = short_model_name[:-5]


d = "train_real_" + str(p)
DatasetCatalog.register("KITTI_MOTS_" + d, 
    lambda d=d: get_KITTI_MOTS_dicts("/home/mcv/datasets/KITTI-MOTS/training/image_02/", train_seqmap))
MetadataCatalog.get("KITTI_MOTS_" + d).set(thing_classes=class_list)

d = "train_clone"
DatasetCatalog.register("KITTI_MOTS_" + d, 
    lambda d=d: get_vKITTI_dicts("/home/mcv/datasets/vKITTI/"))
MetadataCatalog.get("KITTI_MOTS_" + d).set(thing_classes=class_list)

d = "val"
DatasetCatalog.register("KITTI_MOTS_" + d, 
    lambda d=d: get_KITTI_MOTS_dicts("/home/mcv/datasets/KITTI-MOTS/training/image_02/", val_seqmap))
MetadataCatalog.get("KITTI_MOTS_" + d).set(thing_classes=class_list)

d = "test"
DatasetCatalog.register("KITTI_MOTS_" + d, 
    lambda d=d: get_KITTI_MOTS_dicts("/home/mcv/datasets/KITTI-MOTS/training/image_02/", test_seqmap))
MetadataCatalog.get("KITTI_MOTS_" + d).set(thing_classes=class_list)

KITTI_metadata_real = MetadataCatalog.get("KITTI_MOTS_train_real")
KITTI_metadata_clone = MetadataCatalog.get("KITTI_MOTS_train_clone")

for elem in ('first', 'second', 'third'):
    os.makedirs("../" + short_model_name + "_" + elem + "/ground_truth", exist_ok=True)
    os.makedirs("../" + short_model_name + "_" + elem + "/masks", exist_ok=True)
    os.makedirs("../" + short_model_name + "_" + elem + "/inference_val", exist_ok=True)
    os.makedirs("../" + short_model_name + "_" + elem + "/inference_test", exist_ok=True)
    os.makedirs("../" + short_model_name + "_" + elem + "/output", exist_ok=True)
    os.makedirs("../" + short_model_name + "_" + elem + "/output/validation", exist_ok=True)
    os.makedirs("../" + short_model_name + "_" + elem + "/output/test", exist_ok=True)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)  # Let training initialize from model zoo
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model
    cfg.OUTPUT_DIR = "../" + short_model_name + "_" + elem + "/output"
    if elem == 'first':
        cfg.DATASETS.TRAIN = ("KITTI_MOTS_train_real",)
    elif elem == 'second':
        cfg.DATASETS.TRAIN = ("KITTI_MOTS_train_clone",)
    elif elem == 'third':
        cfg.DATASETS.TRAIN = ("KITTI_MOTS_train_real", "KITTI_MOTS_train_clone",)

    cfg.DATASETS.TEST = ("KITTI_MOTS_val",)
    cfg.TEST.EVAL_PERIOD = 800
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025 
    cfg.SOLVER.MAX_ITER = 8000    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  
    cfg.INPUT.MASK_FORMAT='bitmask'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.6 # to be effective in RetinaNet
    cfg.DATASETS.TEST = ("KITTI_MOTS_val", )
    predictor = DefaultPredictor(cfg)

    dataset_dicts = get_KITTI_MOTS_dicts("/home/mcv/datasets/KITTI-MOTS/training/image_02/", val_seqmap)
    ide = 0
    for d in random.sample(dataset_dicts, 40):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        if elem == 'first':
            v = Visualizer(im[:, :, ::-1],
                            metadata=KITTI_metadata_real, 
                            scale=0.8
            )
        elif elem == 'second' or elem == 'third':
            v = Visualizer(im[:, :, ::-1],
                metadata=KITTI_metadata_clone, 
                scale=0.8
            )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite("../" + short_model_name + "_" + elem + "/inference_val/predicted"+str(ide)+".jpg", v.get_image()[:, :, ::-1])
        ide += 1

    dataset_dicts = get_KITTI_MOTS_dicts("/home/mcv/datasets/KITTI-MOTS/training/image_02/", test_seqmap)
    ide = 0
    for d in random.sample(dataset_dicts, 40):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        if elem == 'first':
            v = Visualizer(im[:, :, ::-1],
                            metadata=KITTI_metadata_real, 
                            scale=0.8
            )
        elif elem == 'second' or elem == 'third':
            v = Visualizer(im[:, :, ::-1],
                metadata=KITTI_metadata_clone, 
                scale=0.8
            )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite("../" + short_model_name + "_" + elem + "/inference_test/predicted"+str(ide)+".jpg", v.get_image()[:, :, ::-1])
        ide += 1

    evaluator = COCOEvaluator("KITTI_MOTS_test", cfg, False, output_dir=os.path.join(cfg.OUTPUT_DIR,'test'))
    val_loader = build_detection_test_loader(cfg, "KITTI_MOTS_test")
    inference_on_dataset(predictor.model, val_loader, evaluator)
    model_trained = predictor.model 
    torch.save(model_trained.state_dict(), 'trained_model_' + elem + '.pth')

