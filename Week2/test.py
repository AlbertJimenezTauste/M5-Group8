# import some common libraries
import numpy as np
import cv2
import random
import glob
import os

img_dir = '/home/mcv/datasets/KITTI/data_object_image_2/mini_train/'

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
                "bbox_mode": 'patata',
                "category_id": classes[anno[0]]
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return(dataset_dicts)



get_KITTI_dicts(img_dir, True)
get_KITTI_dicts(img_dir, False)