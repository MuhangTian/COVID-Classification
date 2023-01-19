from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict
from albumentations.pytorch.transforms import ToTensorV2
from ensemble_boxes import ensemble_boxes_wbf
from image_helper import DataAdapter, show_transform_image
import albumentations as A
import pandas as pd
import numpy as np

def run_wbf(predictions, image_size=512, iou_thr=0.55, skip_box_thr=0.42, weights=None):
    bboxes = []
    confidences = []
    class_labels = []

    for prediction in predictions:
        boxes = [(prediction["boxes"] / image_size).tolist()]
        scores = [prediction["scores"].tolist()]
        labels = [prediction["classes"].tolist()]

        boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(
            boxes,
            scores,
            labels,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        boxes = boxes * (image_size - 1)
        bboxes.append(boxes.tolist())
        confidences.append(scores.tolist())
        class_labels.append(labels.tolist())

    return bboxes, confidences, class_labels

def create_model(classes, img_size, backbone, pretrain=True, drop_rate=0.2):
    """
    create instance of the EfficientDet model
    """
    if 'efficientdet' in backbone:  # if using already registered models
        config = get_efficientdet_config(backbone)
    else:        # if DIY our own model
        efficientdet_model_param_dict[backbone] = {
            'name': backbone, 
            'backbone_name': backbone,
            'backbone_args': {'drop_path_rate': drop_rate}, 
            'num_classes': classes, 
            'url': ''}
        config = get_efficientdet_config(backbone)
        
    config.update({'num_classes': classes})
    config.update({'image_size': (img_size, img_size)})
    
    net = EfficientDet(config, pretrained_backbone=pretrain)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes,
    )
    print('{}'.format(config))   # print config
    return DetBenchTrain(net, config)

def train_transform(img_size=512):
    # TODO: Need more work to add more flexibility in data augmentation techniques
    # currently is default value
    # TODO: Some of our image has depth of 4 (related to normalize)
    # TODO: Check the effects of the transform on dataset first then apply it afterwards
    ''' Apply image transform and augmentation for training set '''
    return A.Compose(
        [
        A.Resize(height=img_size, width=img_size, p=1),
        A.Normalize(),
        ToTensorV2(p=1)
        ],
        p=1.0,
        bbox_params=A.BboxParams(format="pascal_voc", 
                                 label_fields=["class_labels"]),
    )

def val_transform(img_size=512):    # CHECK: may need another for test set
    # TODO: Need more work to add more flexibility in data augmentation techniques
    # currently is default value
    # TODO: Some of our image has depth of 4 (in RGBA format)
    # TODO: Check the effects of the transform on dataset first then apply it afterwards
    ''' Apply image transform and augmentation for validation set or validation set '''
    return A.Compose(
        [
        A.Resize(height=img_size, width=img_size, p=1), 
        A.Normalize(),
        ToTensorV2(p=1)
        ],
        p=1.0,
        bbox_params=A.BboxParams(format="pascal_voc", 
                                 label_fields=["class_labels"]),
    )

if __name__ == '__main__':
    transform = A.Compose(
        [
        A.Resize(height=1024, width=1024, p=1),
        A.Normalize(),
        # A.RandomBrightnessContrast(p=1),
        # A.RandomGamma(p=1),
        ],
        p=1.0,
        bbox_params=A.BboxParams(format="pascal_voc", 
                                 label_fields=["class_labels"]),
    )
    
    da = DataAdapter(pd.read_csv('data/self-data.csv'))
    for _ in range(100):
        try: show_transform_image(transform, da, np.random.randint(0, 1500))
        except: pass
    
    # print(list(efficientdet_model_param_dict.keys())[::3])