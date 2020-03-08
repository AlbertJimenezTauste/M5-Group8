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
    image_paths = sorted(glob.glob(img_dir + '*.png'))
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
        """
        if not os.path.isfile('./output/validation/valid_files.txt'):
            with open('./output/validation/valid_files.txt', "w")  as text_file:
                for i in ids:
                    print(label_paths[i], file=text_file)
        """


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
                "bbox": [float(anno[4]), float(anno[5]), float(anno[6]), float(anno[7])],
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
    lambda d=d: get_KITTI_dicts("/home/mcv/datasets/KITTI/data_object_image_2/training/image_2/", True))
MetadataCatalog.get("KITTI_" + d).set(thing_classes=class_list)

d = "val"
DatasetCatalog.register("KITTI_" + d, 
    lambda d=d: get_KITTI_dicts("/home/mcv/datasets/KITTI/data_object_image_2/training/image_2/", False))
MetadataCatalog.get("KITTI_" + d).set(thing_classes=class_list)

KITTI_metadata = MetadataCatalog.get("KITTI_train")
print(KITTI_metadata)


dataset_dicts = get_KITTI_dicts("/home/mcv/datasets/KITTI/data_object_image_2/training/image_2/", True)
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=KITTI_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite("img.jpg", vis.get_image()[:, :, ::-1])


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("KITTI_train",)
cfg.DATASETS.TEST = ("KITTI_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9  # only has one class (ballon)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set the testing threshold for this model
cfg.DATASETS.TEST = ("KITTI_val", )
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_KITTI_dicts("/home/mcv/datasets/KITTI/data_object_image_2/training/image_2/", False)
for d in random.sample(dataset_dicts, 2):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                    metadata=KITTI_metadata, 
                    scale=0.8
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("predicted.jpg", v.get_image()[:, :, ::-1])


for d in dataset_dicts:    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)

    img_name = d["file_name"]
    splitd = img_name.split(os.sep)
    img_name = splitd[-1]
    img_id = img_name.split('.')[0]

    with open('./output/validation/'+ img_id + '.txt', "w") as text_file:
        instances = outputs["instances"].to("cpu")
        print(len(instances))
        for i in range( 0, len(instances) ):
            print(float(instances.pred_boxes.tensor[0][3]))

            print(class_list[instances.pred_classes[i]])
            print(class_list[instances.pred_classes[i]] + ' 0 0 0 ' + str(float(instances.pred_boxes.tensor[i][0])) + ' ' + str(float(instances.pred_boxes.tensor[i][1])) + ' ' + str(float(instances.pred_boxes.tensor[i][2])) + ' ' + str(float(instances.pred_boxes.tensor[i][3])) + ' 0 0 0 0 0 0 0 ' + str(float(instances.scores[i])))
            print(class_list[instances.pred_classes[i]] + ' 0 0 0 ' + str(float(instances.pred_boxes.tensor[i][0])) + ' ' + str(float(instances.pred_boxes.tensor[i][1])) + ' ' + str(float(instances.pred_boxes.tensor[i][2])) + ' ' + str(float(instances.pred_boxes.tensor[i][3])) + ' 0 0 0 0 0 0 0 ' + str(float(instances.scores[i])), file=text_file)


# Instances.um_instances=1, image_height=370, image_width=1224, fields=[pred_boxes: Boxes(tensor([[742.3697, 151.7529, 808.2717, 291.6360]])), scores: tensor([0.9724]), pred_classes: tensor([3])])
# [class 0 0 0 bbox bbox bbox bbox 0 0 0 0 0 0 0 score]
# Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01