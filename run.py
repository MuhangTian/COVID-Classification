from models.EfficientDet import EfficientDetDataModule, EfficientDetModel
from utils.model_helper import train_transform, val_transform
from pytorch_lightning import Trainer
from argparse import ArgumentParser
import wandb
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
def run(config, wdb, mode):
    if wdb == True:
        run = wandb.init(
            entity='muhang-tian',
            project=config['project'],
            config=config)
        config = wandb.config
        
    # --------------------- get hyperparameters based on mode ---------------------
    if mode == 'train':
        img_size = config['img_size']
        path = config['path']
        batch_size = config['batch_size']
        backbone = config['backbone']
        predict_confidence_thres = config['predict_confidence_thres']
        iou_thres = config['iou_thres']
        lr = config['lr']
        optimizer = config['optimizer']
        max_epochs = config['max_epochs']
    elif mode == 'grid':
        img_size = wandb.config.img_size
        path = wandb.config.path
        batch_size = wandb.config.batch_size
        backbone = wandb.config.backbone
        predict_confidence_thres = wandb.config.predict_confidence_thres
        iou_thres = wandb.config.iou_thres
        lr = wandb.config.lr
        optimizer = wandb.config.optimizer
        max_epochs = wandb.config.max_epochs
    else: raise ValueError('Only train or grid allowed as mode')
    # -----------------------------------------------------------------------------
    
    df = pd.read_csv(path)
    module = EfficientDetDataModule( # CHECK: may need to implement test_loader() for test set inside this class
        df=df,
        frac=0.75,
        train_transforms=train_transform(img_size),
        val_transforms=val_transform(img_size),
        batch_size=batch_size)
    
    model = EfficientDetModel(
                            inference_transforms=val_transform(img_size), 
                            backbone=backbone,
                            num_classes=2,
                            img_size=img_size,
                            predict_confidence_thres=predict_confidence_thres,
                            lr=lr,
                            iou_thres=iou_thres,
                            optimizer=optimizer,
    )
    trainer = Trainer(
                    accelerator='auto',
                    max_epochs=max_epochs, 
                    val_check_interval=1,
                    num_sanity_val_steps=1,
                    log_every_n_steps=50,
                    strategy='ddp',      # NOTE: for GPUs in cluster
                    devices=4,         # NOTE: for GPUS in cluster
    )
    
    trainer.fit(model, datamodule=module)
    
    if mode == 'train':
        torch.save(model.state_dict(), f"trained/EfficientDet/{config['backbone']}")
        
    return print('===================== DONE =====================')

if __name__ == '__main__':
    parser = ArgumentParser(description="Parser for EfficientDet Experiments")
    parser.add_argument('-p', dest='config_path', type=str,
                        default='sweep/efficientdetv2_ds.yaml')
    parser.add_argument('-wdb', dest='wdb', type=bool, default=True)
    parser.add_argument('-mode', dest='mode', type=str, default='grid')
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
        
    run(config, args.wdb, args.mode)
    
    
    
    