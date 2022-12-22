import os
import pandas as pd
import numpy as np
from pathlib import Path
import PIL
from PIL import ExifTags
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.config.model_config import efficientdet_model_param_dict
from effdet.efficientdet import HeadNet
import timm
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.core.decorators import auto_move_data    # TODO: Cannot find this
from fastcore.dispatch import typedispatch
from ensemble_boxes import ensemble_boxes_wbf

from numbers import Number
from typing import List
from functools import singledispatch

class dataAdaptor:
    def __init__(self, dir_path, coord_df):
        self.dir_path = Path(dir_path)
        self.coord_df = coord_df
        self.img_names = self.coord_df.name.unique().tolist()
    
    def __len__(self) -> int:
        return len(self.img_names)

    def get_image_and_labels_by_idx(self, index):
        image_name = self.img_names[index]
        image = PIL.Image.open(self.dir_path / image_name)
        pascal_bboxes = self.coord_df[self.coord_df.name==image_name]["xmin","ymin","xmax","ymax"].values
        labels = np.ones(len(pascal_bboxes))
        return image, pascal_bboxes, labels, index

def create_model(num_classes=1, image_size=512, architecture='tf_efficientnetv2_l'):
    efficientdet_model_param_dict['tf_efficientnetv2_l'] = dict(name='tf_efficientnetv2_l', backbone_name='tf_efficientnetv2_l', backbone_args=dict(drop_path_rate=0.2), num_classes=num_classes, url='')
    config = get_efficientdet_config(architecture)
    config.update({'num_classes':num_classes})
    config.update({'image_size':(image_size, image_size)})
    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(config,num_outputs=config.num_classes,)
    return DetBenchTrain(net, config)

def get_train_transforms(target_img_size=512):
    return A.Compose([A.HorizontalFlip(p=0.5), A.Resize(height=target_img_size, width=target_img_size, p=1), A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2(p=1)], p=1.0, bbox_params=A.BboxParams(format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]))

def get_valid_transforms(target_img_size=512):
    return A.Compose([A.Resize(height=target_img_size, width=target_img_size, p=1), A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ToTensorV2(p=1)], p=1.0, bbox_params=A.BboxParams(format="pascal_voc", min_area=0, min_visibility=0, label_fields=["labels"]))

class EfficientDetDataset(Dataset):
    def __init__(self, dataset_adaptor, transforms=get_valid_transforms()):
        self.ds = dataset_adaptor
        self.transforms = transforms

    def __getitem__(self, index):
        (image, pascal_bboxes, class_labels, image_id) = self.ds.get_image_and_labels_by_idx(index)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if 274 in exif:
            if exif[orientation] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image = image.rotate(90, expand=True)
        sample = {"image":np.array(image, dtype=np.float32), "bboxes":pascal_bboxes, "labels":class_labels}
        sample = self.transforms(**sample)
        sample["bboxes"] = np.array(sample["bboxes"])
        image = sample["image"]
        pascal_bboxes = sample["bboxes"]
        labels = sample["labels"]
        _, new_h, new_w = image.shape
        sample["bboxes"][:,[0, 1, 2, 3]] = sample["bboxes"][:,[1, 0, 3, 2]] # convert to yxyx
        target = {"bboxes":torch.as_tensor(sample["bboxes"], dtype=torch.float32), "labels":torch.as_tensor(labels), "image_id":torch.tensor([image_id]), "img_size":(new_h, new_w), "img_scale":torch.tensor([1.0])}
        return image, target, image_id

class EfficientDetDataModule(LightningDataModule):
    def __init__(self,train_dataset_adaptor, validation_dataset_adaptor, train_transforms=get_train_transforms(target_img_size=512), valid_transforms=get_valid_transforms(target_img_size=512), num_workers=4, batch_size=8):
        self.train_ds = train_dataset_adaptor
        self.valid_ds = validation_dataset_adaptor
        self.train_tfms = train_transforms
        self.valid_tfms = valid_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        super().__init__()

    def train_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(dataset_adaptor=self.train_ds, transforms=self.train_tfms)

    def train_dataloader(self) -> DataLoader:
        train_dataset = self.train_dataset()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=self.num_workers, collate_fn=self.collate_fn)
        return train_loader

    def val_dataset(self) -> EfficientDetDataset:
        return EfficientDetDataset(dataset_adaptor=self.valid_ds, transforms=self.valid_tfms)

    def val_dataloader(self) -> DataLoader:
        valid_dataset = self.val_dataset()
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=True, num_workers=self.num_workers, collate_fn=self.collate_fn)
        return valid_loader
    
    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()
        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()
        annotations = {"bbox":boxes, "cls":labels, "img_size":img_size, "img_scale":img_scale}
        return images, annotations, targets, image_ids

def run_wbf(predictions, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    bboxes = []
    confidences = []
    class_labels = []
    for prediction in predictions:
        boxes = [(prediction["boxes"] / image_size).tolist()]
        scores = [prediction["scores"].tolist()]
        labels = [prediction["classes"].tolist()]
        boxes, scores, labels = ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        boxes = boxes * (image_size - 1)
        bboxes.append(boxes.tolist())
        confidences.append(scores.tolist())
        class_labels.append(labels.tolist())
    return bboxes, confidences, class_labels

# build model
class EfficientDetModel(LightningModule):
    def __init__(self, num_classes=1, img_size=512, prediction_confidence_threshold=0.2, learning_rate=2*1e-4,
                 wbf_iou_threshold=0.44, inference_transforms=get_valid_transforms(target_img_size=512),
                 model_architecture='tf_efficientnetv2_l'):
        super().__init__()
        self.img_size = img_size
        self.model = create_model(num_classes, img_size, architecture=model_architecture)
        self.prediction_confidence_threshold = prediction_confidence_threshold
        self.lr = learning_rate
        self.wbf_iou_threshold = wbf_iou_threshold
        self.inference_tfms = inference_transforms

    #@auto_move_data
    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        images, annotations, _, image_ids = batch
        losses = self.model(images, annotations)
        logging_losses = {"class_loss":losses["class_loss"].detach(), "box_loss":losses["box_loss"].detach()}
        self.log("train_loss", losses["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_class_loss", losses["class_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_box_loss", losses["box_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return losses["loss"]

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        outputs = self.model(images, annotations)
        detections = outputs["detections"]
        batch_predictions = {"predictions":detections, "targets":targets, "image_ids":image_ids}
        logging_losses = {"class_loss":outputs["class_loss"].detach(), "box_loss":outputs["box_loss"].detach()}
        self.log("valid_loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("valid_class_loss", logging_losses["class_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("valid_box_loss", logging_losses["box_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return {"loss":outputs["loss"], "batch_predictions":batch_predictions}
    
    @typedispatch
    def predict(self, images: List):
        """
        For making predictions from images
        images: a list of PIL images
        returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences
        """
        image_sizes = [(image.size[1], image.size[0]) for image in images]
        images_tensor = torch.stack([self.inference_tfms(image=np.array(image, dtype=np.float32), labels=np.ones(1), bboxes=np.array([[0, 0, 1, 1]]))["image"] for image in images])
        return self._run_inference(images_tensor, image_sizes)

    @typedispatch
    def predict(self, images_tensor: torch.Tensor):
        """
        For making predictions from tensors returned from the model's dataloader
        images_tensor: the images tensor returned from the dataloader
        returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences
        """
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(0)
        if (images_tensor.shape[-1] != self.img_size or images_tensor.shape[-2] != self.img_size):
            raise ValueError(f"Input tensors must be of shape (N, 3, {self.img_size}, {self.img_size})")
        num_images = images_tensor.shape[0]
        image_sizes = [(self.img_size, self.img_size)] * num_images
        return self._run_inference(images_tensor, image_sizes)

    def _run_inference(self, images_tensor, image_sizes):
        dummy_targets = self._create_dummy_inference_targets(num_images=images_tensor.shape[0])
        detections = self.model(images_tensor.to(self.device), dummy_targets)["detections"]
        (predicted_bboxes, predicted_class_confidences, predicted_class_labels) = self.post_process_detections(detections)
        scaled_bboxes = self.__rescale_bboxes(predicted_bboxes=predicted_bboxes, image_sizes=image_sizes)
        return scaled_bboxes, predicted_class_labels, predicted_class_confidences
    
    def _create_dummy_inference_targets(self, num_images):
        dummy_targets = {"bbox":[torch.tensor([[0.0,0.0,0.0,0.0]], device=self.device) for i in range(num_images)], "cls":[torch.tensor([1.0], device=self.device) for i in range(num_images)], "img_size":torch.tensor([(self.img_size, self.img_size)] * num_images, device=self.device).float(), "img_scale":torch.ones(num_images, device=self.device).float()}
        return dummy_targets
    
    def post_process_detections(self, detections):
        predictions = []
        for i in range(detections.shape[0]):
            predictions.append(self._postprocess_single_prediction_detections(detections[i]))
        predicted_bboxes, predicted_class_confidences, predicted_class_labels = run_wbf(predictions, image_size=self.img_size, iou_thr=self.wbf_iou_threshold)
        return predicted_bboxes, predicted_class_confidences, predicted_class_labels

    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections.detach().cpu().numpy()[:,:4]
        scores = detections.detach().cpu().numpy()[:,4]
        classes = detections.detach().cpu().numpy()[:,5]
        indexes = np.where(scores > self.prediction_confidence_threshold)[0]
        boxes = boxes[indexes]
        return {"boxes":boxes, "scores":scores[indexes], "classes":classes[indexes]}

    def __rescale_bboxes(self, predicted_bboxes, image_sizes):
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_h, im_w = img_dims
            if len(bboxes) > 0:
                scaled_bboxes.append((np.array(bboxes) * [im_w/self.img_size, im_h/self.img_size, im_w/self.img_size, im_h/self.img_size]).tolist())
            else:
                scaled_bboxes.append(bboxes)
        return scaled_bboxes

df = pd.read_csv('FOCUS_train_target_coord.csv')
train_d = dataAdaptor('FOCUS_object_train', df)
dm = EfficientDetDataModule(train_dataset_adaptor=train_d, validation_dataset_adaptor=train_d, num_workers=4, batch_size=2)
model = EfficientDetModel(num_classes=1, img_size=512).to('cuda')
trainer = Trainer(gpus=[0], max_epochs=10, num_sanity_val_steps=1)
trainer.fit(model, dm.to('cuda'))
torch.save(model.state_dict(), 'FOCUS_trained_effdet.pt')

model.eval()
model.to('cpu')
df_test = []
test_path = 'FOCUS_object_test'
dir_list = os.listdir(test_path)
for i, f_name in enumerate(dir_list):
    predicted_bbox, _, _ = model.predict([PIL.Image.open(test_path / f_name)])
    xmin, ymin, xmax, ymax = predicted_bbox[0]
    df_test.append([f_name, xmin, ymin, xmax, ymax])
df_test = pd.DataFrame(df_test, columns=['name','xmin','ymin','xmax','ymax'])
df_test.to_csv('FOCUS_test_target_coord.csv', index=False)
