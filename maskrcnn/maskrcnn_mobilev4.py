from maskrcnn.mobilenetv4 import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import MaskRCNN

def get_mobilev4_model_instance_segmentation(backbone_name, num_classes):

    backbone = MobileNetV4(backbone_name)

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128), (64, 128, 256), (128, 256, 512), (256, 512, 1024)),  # Kích thước anchors cho từng feature map
        aspect_ratios=((0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0)),  # Tỷ lệ khung cho từng feature map
    )
    model = MaskRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


