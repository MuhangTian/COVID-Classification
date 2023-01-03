from models.EfficientDet import EfficientDetDataModule, EfficientDetModel
from utils.model_helper import train_transform, val_transform
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
import torch
import yaml
import pandas as pd

def train(config, wandb, name):
    if wandb == True:
        log = WandbLogger(
        project=config['project'],
        name=name
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
            val_check_interval=0.2,
            num_sanity_val_steps=2,
            log_every_n_steps=20,
        )
    trainer.fit(model, datamodule=module)
    torch.save(model.state_dict(), f"trained/EfficientDet/{config['backbone']}")
    return print('===================== DONE =====================')

if __name__ == '__main__':
    parser = ArgumentParser(description="basic parser for bandit problem")
    parser.add_argument('-p', dest='config_path', type=str,
                        default='config/tf_efficientnetv2_l.yaml')
    parser.add_argument('-wdb', dest='wandb', type=bool, default=False)
    parser.add_argument('-n', dest='name', type=str, default='First Trial')
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)
    train(config, args.wandb, args.name)
    # torch.save([1], f"trained/EfficientDet/{config['backbone']}")
    
    