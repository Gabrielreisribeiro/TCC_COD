from collections import defaultdict
import xmltodict
import pprint
import json
import shutil
import os
import glob

dir_list = list()
lables = ['person', 'bike', 'vehicle', 'motobike', 'airplane', 'bus', 'train', 'truck', 'traffic sign', 'traffic light',
          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
          'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
          'hair drier', 'toothbrush']


def change_extension(file_name):
    dir_name = os.path.dirname(file_name)
    file_name = os.path.basename(file_name)
    file_name = os.path.splitext(file_name)[0]  # discard extension
    return file_name


def listlen(object_name):
    for l in range(len(lables)):
        if (lables[l] == object_name):
            object_name = l
            return object_name

for imageName in glob.glob('/home/gabriel/CarlaPredict/Tcc_Gabriel/venv/Carla-Object-Detection-Dataset/Carla-Datasets/train/*.xml'):
    with open(imageName, 'r', encoding='utf-8') as file:
        my_xml = file.read()
        dictionary = xmltodict.parse(my_xml)
        dir_list.append(dictionary)
        filename = change_extension(imageName)

        # json_object = json.dumps(dictionary, indent=2)
namePath = '/home/gabriel/CarlaPredict/Tcc_Gabriel/venv/yolov7/Correct'
if not os.path.exists(namePath):
    os.makedirs(namePath)
    os.chdir(namePath)

with open(namePath + filename + ".txt", 'w') as fw:
    for i in range(len(dir_list)):
        filename = change_extension(dir_list[i]['annotation']['filename'])
        if ('object' in dir_list[i]['annotation']):
            typeu = type(dir_list[i]['annotation']['object'])
            if (typeu == list):
                for x in range(len(dir_list[i]['annotation']['object'])):
                    frases = list()
                    labels = list()
                    object_name = dir_list[i]['annotation']['object'][x]['name']
                    bndbox_xmin = dir_list[i]['annotation']['object'][x]['bndbox']['xmin']
                    bndbox_ymin = dir_list[i]['annotation']['object'][x]['bndbox']['ymin']
                    bndbox_xmax = dir_list[i]['annotation']['object'][x]['bndbox']['xmax']
                    bndbox_ymax = dir_list[i]['annotation']['object'][x]['bndbox']['ymax']
                    # index = listlen(object_name)
                    bdx = (bndbox_xmin + " " + bndbox_ymin + " " + bndbox_xmax + " " + bndbox_ymax)
                    labels.append(object_name + "\n")
                    frases.append(bdx + "\n")
                    open(namePath + '/' + filename + ".txt", 'a').writelines(frases)
                    open(namePath + '/' + filename + "_labels" + ".txt", 'a').writelines(labels)
            elif typeu == dict:
                for x in range(len(dir_list[i]['annotation']['object'])):
                    frases = list()
                    labels = list()
                    object_name = dir_list[i]['annotation']['object']['name']
                    bndbox_xmin = dir_list[i]['annotation']['object']['bndbox']['xmin']
                    bndbox_ymin = dir_list[i]['annotation']['object']['bndbox']['ymin']
                    bndbox_xmax = dir_list[i]['annotation']['object']['bndbox']['xmax']
                    bndbox_ymax = dir_list[i]['annotation']['object']['bndbox']['ymax']
                    index = listlen(object_name)
                    bdx = (bndbox_xmin + " " + bndbox_ymin + " " + bndbox_xmax + " " + bndbox_ymax)
                    # frases.append(str(index) + " ")
                    labels.append(object_name + "\n")
                    frases.append(bdx)
                    frases.append("\n")
                open(namePath + '/' + filename + ".txt", 'a').writelines(frases)
                open(namePath + '/' + filename + "_labels" + ".txt", 'a').writelines(labels)