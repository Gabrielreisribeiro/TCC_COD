import numpy as np
from torchvision.ops import complete_box_iou_loss, box_iou
from torchvision.io.image import read_image, ImageReadMode
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import math
from torchmetrics import MeanSquaredError
from sklearn.metrics import mean_absolute_error
import os
import pprint
import json as JSON
import re
import torch
import glob
import timeit
import shutil

path_images = "D:/CarlaPredict/Tcc_Gabriel/venv/INRIA/data/set00/extracted_data/set00/V000/images/"
path_annotations = "D:/CarlaPredict/Tcc_Gabriel/venv/INRIA/data/set00/extracted_data/set00/V000/annotations/"
name_path_result = 'ResultadoNovoMAE'
imageYoloPredict = 'ResultadoNovo/Yolo'
score_threshold = 0.5

def detect_with_model(img):
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.iou = 0.5  # NMS IoU threshold
    model.classes = 0  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    # Inference
    start_time = timeit.default_timer()
    results = model(img)
    final_time = timeit.default_timer() - start_time
    # Results
    results.print()
    # results.save()  # or .show()
    resultsDF = results.pandas().xyxy[0]
    # Seleciona as colunas 'xmin', 'ymin', 'xmax' e 'ymax'
    boxes = resultsDF[['xmin', 'ymin', 'xmax', 'ymax']]
    labels = resultsDF['name']
    # Converte as colunas selecionadas em uma matriz numpy
    boxes_array = boxes.values
    labels_array = labels.values
    caminho = os.path.join("./" + imageYoloPredict, name_image)
    print(caminho)

    results.save(save_dir=caminho)

    return boxes_array,labels_array, 'yolov5s', img, final_time

def change_extension(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


def Write_csv(name_city, qtd_previsto,name_weights):
    csv_list = list()
    caminho = os.path.join("./" + name_path_result, name_city)

    if not os.path.exists("./" + name_path_result):
        os.mkdir("./" + name_path_result)
        os.mkdir(caminho)
    elif not os.path.exists(caminho):
        os.mkdir(caminho)

    with open(caminho + '/DataFrame_CSV_' + name_weights + '.csv', 'w', encoding='utf-8') as fw:

        csv_list.append(
            'Cidades' + ',' + 'tensor_predict' + ',' + 'tensor_correct' + ',' + 'index_predict' + ',' + 'Label_Predict' + ','
            + 'index_correct' + ',' + 'label_Correct' + ',' + 'IOU' + ',' + 'IOU_loss' + ',' + 'MSE' + ',' + 'RMSE' + ',' + 'MAE' +'\n')
        open(caminho + "/DataFrame_CSV_" + name_weights + ".csv", 'a').writelines(csv_list)
        for i in range(len(data_list)):
            csv_result = list()
            csv_result.append(name_image + ',' + data_list[i] + '\n')
            open(caminho + "/DataFrame_CSV_" + name_weights + ".csv", 'a').writelines(csv_result)

    with open(caminho + '/Result_Calc_CSV_' + name_weights + '.csv', 'w', encoding='utf-8') as fw:
        list_array = list()
        list_array.append('Cidades' + ',' + 'Qtd. Itens Previstos' + ',' + 'Qtd. Itens Existentes' + ','
                          + 'Qtd. Itens Previstos Corretamente (iou > 0.5)' + ',' + 'Percentual de acerto' + ','
                          + 'Qtd. Itens Previstos Incorretamente (iou < 0.5)' + ',' + 'Percentual de erro' + ','
                          + 'Qtd. Itens Não-Previstos' + ',' + ' MSE_SOMA' + ',' + ' IOU_LOSS_SOMA' + ',' + 'Tempo de Predição' '\n')
        list_array.append(name_image + ',' + str(len(labels)) + ',' + str(len(qtd_previsto)) + ',' + str(
            ItensCorretos) + ','
                          + str(PercentualCorreto) + '%' + ',' + str(ItensIncorretos) + ',' + str(
            PercentualIncorreto) + '%' + ','
                          + str(ItensNãoPrevistos) + ',' + str(MSE_SUM) + ',' + str(
            IOU_loss_SUM) + ',' + str(final_time) + '\n')
        open(caminho + "/Result_Calc_CSV_" + name_weights + ".csv", 'a').writelines(list_array)


def boundry_filter(bdx):
    bdx_filter = []
    for boundbox in bdx:
        xmin = boundbox[0]
        ymin = boundbox[1]
        width = boundbox[2]
        height = boundbox[3]
        xmax = xmin + width
        ymax = ymin + height
        bdx_filter.append([xmin,ymin,xmax,ymax])
    return bdx_filter

def json_read(item):
    name_json = change_extension(item)
    Items_predict = list()
    bdx_predict = list()
    with open(item, 'r') as fw:
        json_informations = JSON.load(fw)
        if (json_informations == []):
            print("array vazio")
        else:
            if (type(json_informations) == list):
                for x in range(len(json_informations)):
                    object_name = json_informations[x]['lbl']
                    bndbox = json_informations[x]['pos']
                    Items_predict.append(object_name)
                    bdx_predict.append(bndbox)
    bdx_predict = boundry_filter(bdx_predict)
    return name_json, Items_predict, bdx_predict

jsonData = []
for jsonItem in glob.glob(path_annotations + "*.json"):
    name_json, object_name, bdx = json_read(jsonItem)
    jsonData.append([name_json, object_name, bdx])

images = glob.glob(path_images + "*.png")
for i in range(len(images)):
    name_image = change_extension(images[i])
    if name_image in jsonData[i]:
        boxes, labels, name_weights, img, final_time = detect_with_model(images[i])  # me devolve o modelo predito com as lebals preditas e os bdx preditos
        count = 0
        MSE_SUM = 0
        IOU_loss_SUM = 0
        data_list = list()
        index_xml_arr = []
        for y in range(len(jsonData[i][1])):
            for x in range(len(labels)):
                tensor_predict = torch.tensor(boxes[x], dtype=torch.float).unsqueeze(0)
                tensor_xml = torch.tensor(jsonData[i][2][y], dtype=torch.float).unsqueeze(0)
                tensor_predict_arr = str(tensor_predict.detach().numpy())
                tensor_xml_arr = str(tensor_xml.detach().numpy())
                IOU = float(box_iou(tensor_xml, tensor_predict))
                IOU_loss = float(complete_box_iou_loss(tensor_predict, tensor_xml))
                mean_squared_error = MeanSquaredError()
                MAE = float(mean_absolute_error(tensor_xml, tensor_predict))
                MSE = float(mean_squared_error(tensor_xml, tensor_predict))
                RMSE = float(math.sqrt(MSE))
                if (IOU > score_threshold):
                    if(y not in index_xml_arr):
                        index_predict = str(x)
                        index_xml = str(y)
                        index_xml_arr.append(y)
                        labels_predict = str(labels[x])
                        labels_xml = str(jsonData[i][1][y])
                        IOU = str(IOU)
                        IOU_loss = str(IOU_loss)
                        MSE = str(MSE)
                        MAE = str(MAE)
                        RMSE = str(RMSE)
                        count += 1
                        MSE_SUM = MSE_SUM + float(MSE)
                        IOU_loss_SUM = IOU_loss_SUM + float(IOU_loss)
                        data_list.append(tensor_predict_arr + ','
                                         + tensor_xml_arr + ','
                                         + index_predict + ','
                                         + labels_predict + ','
                                         + index_xml + ','
                                         + labels_xml + ','
                                         + IOU + ','
                                         + IOU_loss + ','
                                         + MSE + ','
                                         + RMSE + ','
                                         + MAE)
            ItensCorretos = count
            ItensIncorretos = (len(labels) - ItensCorretos)
            ItensNãoPrevistos = (len(jsonData[i][2]) - ItensCorretos)
            if (len(jsonData[i][1]) == 0):
                PercentualCorreto = 0.0
            else:
                PercentualCorreto = (ItensCorretos / len(jsonData[i][1])) * 100
                PercentualIncorreto = 100 - PercentualCorreto
                Write_csv(name_image,jsonData[i][2],name_weights)
