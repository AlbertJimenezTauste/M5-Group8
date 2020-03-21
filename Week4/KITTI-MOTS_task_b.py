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

from detectron2.evaluation import COCOEvaluator, inference_on_dataset, SemSegEvaluator

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

                if MODE == 0:
                    obj = {
                        "bbox": [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": 0 if instance.class_id == 2 else 2
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
                        "category_id": 0 if instance.class_id == 2 else 2,
                        "id": obj_num
                    }

                    obj_num = obj_num+1

                elif MODE == 3:
                    obj = {
                        "bbox": [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": 0 if instance.class_id == 2 else 2,
                        "segmentation": rletools.encode(np.asarray(mask, order="F"))
                    }

                elif MODE == 4:
                    obj = {
                        "bbox": [np.min(pos[1]), np.min(pos[0]), np.max(pos[1]), np.max(pos[0])],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": 0 if instance.class_id == 2 else 2,
                        "iscrowd": 0,
                        "segmentation": instance.mask
                    }
    
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    
    print('Loaded ' + str(np.shape(dataset_dicts)[0]) + ' images!')
    return(dataset_dicts)


# MODEL
# ----------------------------------------------------------------------------
models_COCO = ['COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml', 'COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml', 
'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml', 'COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml',
'COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml', 'COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml', 
'COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml', 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml',
'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml', 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml']

models_CITYSCAPES = ['Cityscapes/mask_rcnn_R_50_FPN.yaml']

first_time = True

for model_name in models_CITYSCAPES:

    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg

    cfg = get_cfg()

    #model_name = "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"
    short_model_name = model_name.split("/")
    short_model_name = short_model_name[1]
    short_model_name = short_model_name[:-5]

    os.makedirs("../" + short_model_name + "/ground_truth", exist_ok=True)
    os.makedirs("../" + short_model_name + "/masks", exist_ok=True)
    os.makedirs("../" + short_model_name + "/inference", exist_ok=True)
    os.makedirs("../" + short_model_name + "/output", exist_ok=True)

    #RetinaNet
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
    #cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.6 # to be effective in RetinaNet

    #Fast R-CNN
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model

    #Fast R-CNN
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)  # Let training initialize from model zoo
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model

    print(model_name)
    cfg.DATASETS.TEST = ("KITTI_MOTS_val",)
    cfg.INPUT.MASK_FORMAT= 'bitmask'
    cfg.OUTPUT_DIR = "../" + short_model_name + "/output"

    '''
    cfg.DATASETS.TRAIN = ("KITTI_MOTS_train",)
    cfg.DATASETS.TEST = ("KITTI_MOTS_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    '''

    '''
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (ballon)
    '''
    """
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    """
    '''
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    cfg.DATASETS.TEST = ("KITTI_MOTS_val", )
    '''

    predictor = DefaultPredictor(cfg)


    from detectron2.data import DatasetCatalog, MetadataCatalog

    # CONFIGURATION FOR TRAINING
    # ----------------------------------------------------------------------------
    '''
    # 0-->None, 1-->Car, 2--->Pedestrian
    class_list =   ['None','Car','Pedestrian']
    # Sequences to load:
    seqmap_train = ['0000','0001','0003','0004','0005','0009','0011','0012','0015','0017','0019','0020']
    seqmap_val = ['0002','0006','0007','0008','0010','0013','0014','0016','0018']

    # Create the dataset
    d = "train"
    DatasetCatalog.register("KITTI_MOTS_" + d, 
        lambda d=d: get_KITTI_MOTS_dicts("/home/mcv/datasets/KITTI-MOTS/training/image_02/", seqmap_train))
    MetadataCatalog.get("KITTI_MOTS_" + d).set(thing_classes=class_list)

    d = "val"
    DatasetCatalog.register("KITTI_MOTS_" + d, 
        lambda d=d: get_KITTI_MOTS_dicts("/home/mcv/datasets/KITTI-MOTS/training/image_02/", seqmap_val))
    MetadataCatalog.get("KITTI_MOTS_" + d).set(thing_classes=class_list)

    KITTI_metadata = MetadataCatalog.get("KITTI_MOTS_train")
    print(KITTI_metadata)
    '''

    if first_time:
        # CONFIGURATION FOR INFERENCE WITH MAPPING CLASSES
        # ----------------------------------------------------------------------------
        seqmap_val =   ['0002','0006','0007','0008','0010','0013','0014','0016','0018']
        class_list =   ['None','Car','Pedestrian']

        d = "val"
        DatasetCatalog.register("KITTI_MOTS_" + d, 
            lambda d=d: get_KITTI_MOTS_dicts("/home/mcv/datasets/KITTI-MOTS/training/image_02/", seqmap_val))
        MetadataCatalog.get("KITTI_MOTS_" + d).set(thing_classes=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)

        KITTI_metadata = MetadataCatalog.get("KITTI_MOTS_val")
        first_time = False
        
    print(KITTI_metadata)


    print("---------------------")
    print(MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes)
    print("---------------------")


    # VERIFY DATASET
    # ----------------------------------------------------------------------------
    print("////Verify dataset////")
    dataset_dicts = get_KITTI_MOTS_dicts("/home/mcv/datasets/KITTI-MOTS/training/image_02/", seqmap_val)
    ide = 0
    for d in random.sample(dataset_dicts, 20):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=KITTI_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite("../" + short_model_name + "/ground_truth/gt_img_"+str(ide)+".jpg", vis.get_image()[:, :, ::-1])
        ide += 1



    # EVALUATION WITH COCO: DETECTION
    # ----------------------------------------------------------------------------
    print("////Evaluation////")
    evaluator = COCOEvaluator("KITTI_MOTS_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "KITTI_MOTS_val")
    inference_on_dataset(predictor.model, val_loader, evaluator)

    '''
    # EVALUATION WITH COCO: SEGMENTATION
    # ----------------------------------------------------------------------------
    print("////Evaluation////")
    evaluator = SemSegEvaluator('KITTI_MOTS_val', distributed=True, num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES, output_dir=cfg.OUTPUT_DIR)
    print("1")
    evaluator.evaluate()
    print("2")
    #evaluator.process(dataset_dicts)
    #val_loader = build_detection_test_loader(cfg, "KITTI_MOTS_val")
    #inference_on_dataset(predictor.model, val_loader, evaluator)
    #evaluator = COCOEvaluator("KITTI_MOTS_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    '''

    # INFERENCE EXAMPLES
    # ----------------------------------------------------------------------------
    print("////Inference examples////")
    dataset_dicts = get_KITTI_MOTS_dicts("/home/mcv/datasets/KITTI-MOTS/training/image_02/", seqmap_val)
    ide = 0
    for d in random.sample(dataset_dicts, 20):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                        metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
                        scale=0.8
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite("../" + short_model_name + "/inference/predicted"+str(ide)+".jpg", v.get_image()[:, :, ::-1])
        ide += 1
