import numpy as np

from torchvision.ops import complete_box_iou_loss, box_iou
from torchvision.io.image import read_image, ImageReadMode
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import math
from torchmetrics import MeanSquaredError
import os
import pprint
import json as JSON
import re
import torch
import glob
import timeit

def compare_model_detect(name_path_json,name_path_image, name_path_result, weights, score_threshold, model):
    data_path = "D:/CarlaPredict/Tcc_Gabriel/venv/INRIA/data/set00/extracted_data/set00/V000/"
    #pega apenas o nome da pasta
    def change_extension (file_path):
        return os.path.splitext(os.path.basename(file_path))[0]

    def detect_with_model(img, score_threshold, weights, model):
        model.eval()
        preprocess = weights.transforms()
        name_weights = str(weights)
        img = read_image(img, ImageReadMode.RGB)
        batch = [preprocess(img)]
        start_time = timeit.default_timer()
        prediction = model(batch)[0]
        final_time = timeit.default_timer() - start_time
        labels_array = []
        boxes_array = []
        labels = prediction['labels'][prediction['scores'] > score_threshold]
        boxes = prediction['boxes'][prediction['scores'] > score_threshold]
        boxes = boxes.detach().numpy()
        for labelsid in labels:
            if labelsid == 1:
                labels_array.append(labelsid)
        for i in range(len(labels)):
            if labels[i] == 1:
                boxes_array.append(boxes[i])
        labels = [weights.meta["categories"][i] for i in labels_array]
        return boxes_array, labels, name_weights, img, final_time

    # def detect_with_model(img, score_threshold, weights, model):
    #     model.eval()
    #     preprocess = weights.transforms()
    #     name_weights = str(weights)
    #     img = read_image(img, ImageReadMode.RGB)
    #     batch = [preprocess(img)]
    #     start_time = timeit.default_timer()
    #     prediction = model(batch)[0]
    #     final_time = timeit.default_timer() - start_time
    #     labels_array = []
    #     boxes_array = []
    #     for label, box, score in zip(prediction["labels"], prediction["boxes"], prediction["scores"]):
    #         if label == 1 and score > score_threshold:
    #             labels_array.append(label)
    #             boxes_array.append(box)
    #     labels = [weights.meta["categories"][i] for i in labels_array]
    #     return boxes_array, labels, name_weights, img, final_time

    def json_read(item):
        name_json = change_extension(item)
        Items_predict = list()
        bdx_predict = list()
        with open(item, 'r') as fw:
            json_informations = JSON.load(fw)
            if(json_informations == []):
                print("array vazio")
            else:
                if(type(json_informations) == list):
                    for x in range(len(json_informations)):
                        object_name = json_informations[x]['lbl']
                        bndbox = json_informations[x]['pos']
                        Items_predict.append(object_name)
                        bdx_predict.append(bndbox)
        bdx_predict = boundry_filter(bdx_predict)
        return name_json,Items_predict, bdx_predict
# '''def json_read(item):
#     with open(item, 'r') as fw:
#         json_informations = JSON.load(fw)
#         if json_informations == []:
#             print("array vazio")
#             return None
#         elif not json_informations:
#             return None
#         else:
#             items_predict = [(x['lbl'], x['pos']) for x in json_informations]
#             return items_predict
# '''

    def Write_csv(name_city,qtd_previsto):
        csv_list = list()
        caminho = os.path.join("./"+name_path_result, name_city)
        if not os.path.exists("./"+name_path_result):
            os.mkdir("./"+name_path_result)
            os.mkdir(caminho)
        elif not os.path.exists(caminho):
            os.mkdir(caminho)
#             '''def write_csv(name_city):
#     csv_list = []
#     result_path = "./" + name_path_result
#     city_path = os.path.join(result_path, name_city)
#     if not os.path.exists(result_path):
#         os.mkdir(result_path)
#         os.mkdir(city_path)
#     elif not os.path.exists(city_path):
#         os.mkdir(city_path)
# '''

        with open(caminho + '/DataFrame_CSV_' + name_weights + '.csv', 'w', encoding='utf-8') as fw:
            csv_list.append(
                'Cidades' + ',' + 'tensor_predict' + ',' + 'tensor_correct' + ',' + 'index_predict' + ',' + 'Label_Predict' + ','
                + 'index_correct' + ',' + 'label_Correct' + ',' + 'IOU' + ',' + 'IOU_loss' + ',' + 'MSE' + ',' + 'RMSE' + ',' + 'MAE' + '\n')
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
            list_array.append(name_image + ',' + str(len(labels)) + ',' + str(len(qtd_previsto)) + ',' + str(ItensCorretos) + ','
                              + str(PercentualCorreto) + '%' + ',' + str(ItensIncorretos) + ',' + str(PercentualIncorreto) + '%' + ','
                              + str(ItensNãoPrevistos) + ',' + str(MSE_SUM) + ',' + str(IOU_loss_SUM) + ',' + str(final_time) + '\n')
            open(caminho + "/Result_Calc_CSV_" + name_weights + ".csv", 'a').writelines(list_array)
# ''''with open(caminho + "/DataFrame_CSV_" + name_weights + ".csv", 'w', encoding='utf-8') as fw:
#     # Construir o cabeçalho do arquivo CSV
#     header = f'Cidades,tensor_predict,tensor_correct,index_predict,Label_Predict,index_correct,label_Correct,IOU,IOU_loss,MSE,RMSE\n'
#     # Escrever o cabeçalho no arquivo
#     fw.write(header)
#     # Escrever os resultados em uma linha por vez
#     for i in range(len(data_list)):
#         fw.write(f'{name_image},{data_list[i]}\n')
#
# with open(caminho + '/Result_Calc_CSV_' + name_weights + '.csv', 'w', encoding='utf-8') as fw:
#     # Construir o cabeçalho do arquivo CSV
#     header = f'Cidades,Qtd. Itens Previstos,Qtd. Itens Existentes,Qtd. Itens Previstos Corretamente (iou > 0.5),Percentual de acerto,Qtd. Itens Previstos Incorretamente (iou < 0.5),Percentual de erro,Qtd. Itens Não-Previstos,Tempo de Predição\n'
#     # Escrever o cabeçalho no arquivo
#     fw.write(header)'''

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

#     '''def boundry_filter(bdx):
#     bdx_filter = []
#     for boundbox in bdx:
#         xmin, ymin, width, height = boundbox
#         xmax = xmin + width
#         ymax = ymin + height
#         bdx_filter.append([xmin, ymin, xmax, ymax])
#     return bdx_filter
# '''

    def Save_to_image(img,name_city,boxes,bdx):
            caminho = os.path.join("./"+name_path_result, name_city)
            # if not(bdx):
            #     print('Não há itens Existentes nos dados originais!')
            # else:
            #     bdx_existentes = torch.tensor(bdx, dtype=torch.float)
            #     box = draw_bounding_boxes(img, bdx_existentes,
            #                               colors="yellow",
            #                               width=4)
            #     box = box.permute(1, 2, 0)
            #     box = box.detach().numpy()
            #     plt.imsave(caminho + '/correct_' + name_weights + '.png', box)

            # if not(boxes):
            #     print('Não há itens Existentes nos dados Preditos!')
            # else:
            #     boxes = torch.tensor(boxes, dtype=torch.float)
            #     print('img',img)
            #     box = draw_bounding_boxes(img, boxes,
            #                               colors="green",
            #                               width=4)
            #     box = box.permute(1, 2, 0)
            #     box = box.detach().numpy()
            #     plt.imsave(caminho + '/predict_' + name_weights + '.png', box)


# ''''def save_to_image(img, name_city, boxes, bdx):
#     # Cria o caminho para salvar a imagem
#     caminho = os.path.join("./"+name_path_result, name_city)
#
#     # Verifica se existem boxes existentes
#     if bdx:
#         bdx_existentes = torch.tensor(bdx, dtype=torch.float)
#         box = draw_bounding_boxes(img, bdx_existentes, colors="yellow", width=4)
#         # Converte o tensor em um array numpy e salva a imagem
#         box = box.permute(1, 2, 0).detach().numpy()
#         plt.imsave(caminho + '/correct_' + name_weights + '.png', box)
#     else:
#         print("Não há itens existentes nos dados originais!")
#
#     # Verifica se existem boxes preditos
#     if boxes:
#         boxes = torch.tensor(boxes, dtype=torch.float)
#         box = draw_bounding_boxes(img, boxes, colors="green", width=4)
#         # Converte o tensor em um array numpy e salva a imagem
#         box = box.permute(1, 2, 0).detach().numpy()
#         plt.imsave(caminho + '/predict_' + name_weights + '.png', box)
#     else:
#         print("Não há itens existentes nos dados preditos!")
# '''



    jsonData = []
    for jsonItem in glob.glob(data_path+name_path_json+"/*.json"):
        name_json,object_name, bdx = json_read(jsonItem)
        jsonData.append([name_json,object_name, bdx])

    images = glob.glob(data_path + name_path_image + "/*.png")
    for i in range(len(images)):
        name_image = change_extension(images[i])
        if name_image in jsonData[i]:
            boxes, labels, name_weights, img, final_time = detect_with_model(images[i], score_threshold, weights,model)  # me devolve o modelo predito com as lebals preditas e os bdx preditos
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
                    MAE = float(mean_absolute_error(tensor_xml, tensor_predict))
                    IOU_loss = float(complete_box_iou_loss(tensor_predict, tensor_xml))
                    mean_squared_error = MeanSquaredError()
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
                            RMSE = str(RMSE)
                            MAE = str(MAE)
                            count += 1
                            MSE_SUM = MSE_SUM + float(MSE)
                            IOU_loss_SUM = IOU_loss_SUM + float(IOU_loss)
                            data_list.append(
                                                tensor_predict_arr + ',' + tensor_xml_arr + ',' + index_predict + ',' + labels_predict + ',' + index_xml + ','
                                                + labels_xml + ',' + IOU + ',' + IOU_loss + ',' + MSE + ',' + RMSE + ',' + MAE)
                ItensCorretos = count
                ItensIncorretos = (len(labels) - ItensCorretos)
                ItensNãoPrevistos = (len(jsonData[i][2]) - ItensCorretos)
                if (len(jsonData[i][1]) == 0):
                    PercentualCorreto = 0.0
                else:
                    PercentualCorreto = (ItensCorretos / len(jsonData[i][1])) * 100
                    PercentualIncorreto = 100 - PercentualCorreto
                    Write_csv(name_image,jsonData[i][2])
                    Save_to_image(img, name_image, boxes, jsonData[i][2])

#
#                     ''''jsonData = []
# for jsonItem in glob.glob(data_path+name_path_json+"/*.json"):
#     name_json,object_name, bdx = json_read(jsonItem)
#     jsonData.append([name_json,object_name, bdx])
#
# images = glob.glob(data_path + name_path_image + "/*.png")
# for i in range(len(images)):
#     name_image = change_extension(images[i])
#     if name_image in jsonData[i]:
#         boxes, labels, name_weights, img, final_time = detect_with_model(images[i], score_threshold, weights,model)
#
#         data_list = []
#         for y in range(len(jsonData[i][1])):
#             for x in range(len(labels)):
#                 tensor_predict = torch.tensor(boxes[x], dtype=torch.float).unsqueeze(0)
#                 tensor_xml = torch.tensor(jsonData[i][2][y], dtype=torch.float).unsqueeze(0)
#
#                 IOU = box_iou(tensor_xml, tensor_predict)
#                 IOU_loss = complete_box_iou_loss(tensor_predict, tensor_xml)
#                 mean_squared_error = MeanSquaredError()
#                 MSE = mean_squared_error(tensor_xml, tensor_predict)
#                 RMSE = math.sqrt(MSE)
#
#                 if (IOU > score_threshold):
#                     data_list.append([tensor_predict, tensor_xml, x, labels[x], y, jsonData[i][1][y], IOU, IOU_loss, MSE, RMSE])
#                     count += 1
#
#         ItensCorretos = count
#         ItensIncorretos = (len(labels) - ItensCorretos)
#         ItensNãoPrevistos = (len(jsonData[i][1]) - ItensCorretos)
#
#         if (len(jsonData[i][1]) == 0):
#             PercentualCorreto = 0.0
#         else:
#             PercentualCorreto = (ItensCorretos / len(jsonData[i][1])) * 100
#             PercentualIncorreto = 100 - PercentualCorreto
#             Write_csv(name_image)
#             print(jsonData[i][2])
#             Save_to_image(img, name_image, boxes, jsonData[i][2])
# '''