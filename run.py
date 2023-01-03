from models.EfficientDet import EfficientDetDataModule, EfficientDetModel
from utils.model_helper import train_transform, val_transform
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
import torch
import yaml
import pandas as pd

'''
list of available models from effdet library:

['efficientdet_d0', 'efficientdet_d3', 'efficientdetv2_dt', 'cspresdet50', 
'cspdarkdet53', 'mixdet_l', 'mobiledetv3_large', 'efficientdet_q2', 
'efficientdet_em', 'tf_efficientdet_d1', 'tf_efficientdet_d4', 
'tf_efficientdet_d7', 'tf_efficientdet_d1_ap', 'tf_efficientdet_d4_ap', 
'tf_efficientdet_lite1', 'tf_efficientdet_lite3x']

For more details: https://github.com/rwightman/efficientdet-pytorch
'''
def train(config, wandb):
    if wandb == True:
        log = WandbLogger(
        project=config['project'],
        name=config['name']
        )
    else: log = True
    img_size = config['img_size']
    df = pd.read_csv(config['path'])
    
    module = EfficientDetDataModule( # CHECK: may need to implement test_loader() for testing inside this class
        df=df,
        frac=0.75,
        train_transforms=train_transform(img_size),
        val_transforms=val_transform(img_size),
        batch_size=config['batch_size'])
    
    model = EfficientDetModel(inference_transforms=val_transform(img_size), 
                              backbone=config['backbone'],
                              num_classes=2,
                              img_size=img_size,
                              predict_confidence_thres=config['predict_confidence_thres'],
                              lr=config['lr'],
                              iou_thres=config['iou_thres']
                              )
    trainer = Trainer(
            logger=log, 
            accelerator='auto',
            devices='auto',
            max_epochs=config['max_epochs'], 
            val_check_interval=1,
            num_sanity_val_steps=2,
            log_every_n_steps=20,
        )
    trainer.fit(model, datamodule=module)
    torch.save(model.state_dict(), f"trained/EfficientDet/{config['backbone']}")
    return print('===================== DONE =====================')

if __name__ == '__main__':
    parser = ArgumentParser(description="parser for EfficientDet")
    parser.add_argument('-p', dest='config_path', type=str,
                        default='config/efficientnetv2_ds.yaml')
    parser.add_argument('-wdb', dest='wandb', type=bool, default=False)
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    train(config, args.wandb)
    
    