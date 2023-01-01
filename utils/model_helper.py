from effdet.config.model_config import efficientdet_model_param_dict
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict
from albumentations.pytorch.transforms import ToTensorV2
from image_helper import DataAdapter, get_path, show_image, show_transform_image
import albumentations as A
import numpy as np
import cv2

def create_model(classes, img_size, backbone, pretrain=True, drop_rate=0.2):
    """
    create instance of the EfficientDet model
    """
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
    print('Model created, config:\n{}'.format(config))
    return DetBenchTrain(net, config)

def train_transform(img_size=512):
    # TODO: Need more work to add more flexibility in data augmentation techniques
    # currently is default value
    # TODO: Some of our image has depth of 4 (related to normalize)
    # TODO: Check the effects of the transform on dataset first then apply it afterwards
    return A.Compose(
        [A.HorizontalFlip(p=0.5),
         A.Resize(height=img_size, width=img_size, p=1),
         A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
         ToTensorV2(p=1)],
        p=1.0,
        bbox_params=A.BboxParams(format="pascal_voc", 
                                 label_fields=["class_labels"]),
    )

def test_transform(img_size=512):
    # TODO: Need more work to add more flexibility in data augmentation techniques
    # currently is default value
    # TODO: Some of our image has depth of 4 (related to normalize)
    # TODO: Check the effects of the transform on dataset first then apply it afterwards
    ''' For validation set or test set '''
    return A.Compose(
        [A.Resize(height=img_size, width=img_size, p=1), 
         A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
         ToTensorV2(p=1)],
        p=1.0,
        bbox_params=A.BboxParams(format="pascal_voc", 
                                 label_fields=["class_labels"]),
    )

if __name__ == '__main__':
    transform = A.Compose(
        [A.HorizontalFlip(p=0.5),
         A.Resize(height=512, width=512, p=1),
         A.Normalize([0.485, 0.456, 0.406, 0.3], [0.229, 0.224, 0.225, 0.3])],
        p=1.0,
        bbox_params=A.BboxParams(format="pascal_voc", 
                                 label_fields=["class_labels"]),
    )
    DA = DataAdapter('data/self-data.csv')
    for i in range(1, 1400):
        try: show_transform_image(transform, DA, i)
        except: pass
    