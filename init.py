from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_Weights, ssd300_vgg16, \
    SSD300_VGG16_Weights, retinanet_resnet50_fpn,RetinaNet_ResNet50_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn,FasterRCNN_MobileNet_V3_Large_FPN_Weights,retinanet_resnet50_fpn_v2,RetinaNet_ResNet50_FPN_V2_Weights
from compare import compare_model_detect

name_path_json = 'annotations'
name_path_image = 'images'
name_path_result = 'Resultado'
# weights = SSD300_VGG16_Weights.DEFAULT
weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
# weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
#weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
model = retinanet_resnet50_fpn(weights=weights)
# model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
score_threshold = 0.5
compare_model_detect(name_path_json,name_path_image,name_path_result,weights,score_threshold,model)