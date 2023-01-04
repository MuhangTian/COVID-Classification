import sys
sys.path.insert(0, '/home/users/mt361/COVID-Classification')
sys.path.insert(0, '/home/users/mt361/COVID-Classification/utils')
from utils.model_helper import create_model, run_wbf
from utils.image_helper import DataAdapter
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from fastcore.dispatch import typedispatch
from objdetecteval.metrics.coco_metrics import get_coco_stats #pip install git+https://github.com/alexhock/object-detection-metrics
from typing import List
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd

class EfficientDetDataset(Dataset):
    """
    Hold dataset using adapter in order to use PyTorch

    Args:
        da : DataAdapter
        transforms: function which applies data transform (e.g. train_trainsform)
    """
    def __init__(self, da, transforms):
        self.da = da
        self.transforms = transforms

    def __getitem__(self, idx):
        (name, img, bbox, label) = self.da.i_get(idx)
        sample = {
            "image": np.array(img, dtype=np.float32),
            "bboxes": bbox,
            "class_labels": label,
        }

        transformed = self.transforms(**sample) # Apply data augmentation and transformation
        img = transformed["image"]  # get transformed image
        transformed['bboxes'] = np.array(transformed["bboxes"])
        label = transformed["class_labels"]

        _, height, width = img.shape    # the first is channel (depth), not needed
        transformed["bboxes"][:, [0, 1, 2, 3]] = transformed["bboxes"][:, [1, 0, 3, 2]]  # convert to yxyx
        target = {
            "bboxes": torch.as_tensor(transformed["bboxes"], dtype=torch.float32),
            "labels": torch.as_tensor(label),
            "image_id": torch.tensor([name]),
            "img_size": (height, width),
            "img_scale": torch.tensor([1.0]),
        }

        return img, target, name

    def __len__(self): return len(self.da)

class EfficientDetDataModule(LightningDataModule):
    ''' https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html '''
    def __init__(self, df, frac, train_transforms,
                 val_transforms, num_workers=4, batch_size=8):
        df_train = df.sample(frac=frac)
        df_val = df.drop(df_train.index)
        self.train_da = DataAdapter(df_train)
        self.val_da = DataAdapter(df_val)
        self.train_tfms = train_transforms
        self.val_tfms = val_transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        super().__init__()

    def train_dataloader(self) -> DataLoader:
        train_dataset = EfficientDetDataset(da=self.train_da, transforms=self.train_tfms)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_dataset = EfficientDetDataset(da=self.val_da, transforms=self.val_tfms)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return val_loader
    
    @staticmethod
    def collate_fn(batch):
        images, targets, image_ids = tuple(zip(*batch))
        images = torch.stack(images)
        images = images.float()

        boxes = [target["bboxes"].float() for target in targets]
        labels = [target["labels"].float() for target in targets]
        img_size = torch.tensor([target["img_size"] for target in targets]).float()
        img_scale = torch.tensor([target["img_scale"] for target in targets]).float()

        annotations = {
            "bbox": boxes,
            "cls": labels,
            "img_size": img_size,
            "img_scale": img_scale,
        }

        return images, annotations, targets, image_ids

class EfficientDetModel(pl.LightningModule):
    def __init__(
        self,
        inference_transforms,
        backbone,
        num_classes=2,
        img_size=512,
        predict_confidence_thres=0.2,
        lr=0.0002,
        iou_thres=0.44,
        optimizer='AdamW'
    ):
        super().__init__()
        self.img_size = img_size
        self.model = create_model(num_classes, img_size, backbone)
        self.predict_confidence_thres = predict_confidence_thres
        self.lr = lr
        self.iou_thres = iou_thres
        self.inference_tfms = inference_transforms
        self.optimizer = optimizer

    def forward(self, images, targets):
        return self.model(images, targets)

    def configure_optimizers(self):
        if self.optimizer == 'AdamW':
            return torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
    
    def training_step(self, batch, batch_idx):
        images, annotations, _, image_ids = batch
        losses = self.model(images, annotations)
        self.log('Train Loss', losses['loss'].detach())
        self.log('Train Classification Loss', losses["class_loss"].detach())
        self.log('Train Localization Loss', losses["box_loss"].detach())
        return losses['loss']
    
    @torch.no_grad()    
    def validation_step(self, batch, batch_idx):
        images, annotations, targets, image_ids = batch
        outputs = self.model(images, annotations)
        detections = outputs["detections"]
        batch_predictions = {
            "predictions": detections,
            "targets": targets,
            "image_ids": image_ids,
        }
        self.log('Validation Loss', outputs["loss"])
        self.log('Validation Classification Loss', outputs["class_loss"].detach())
        self.log('Validation Localization Loss', outputs["box_loss"].detach())
        # predicted_bboxes, predicted_class_confidences, predicted_class_labels = self.predict(images)
        return {'loss': outputs["loss"], 'batch_predictions': batch_predictions}
    
    def validation_epoch_end(self, outputs):
        validation_loss_mean = torch.stack(
            [output["loss"] for output in outputs]
        ).mean()

        (
            predicted_class_labels,
            image_ids,
            predicted_bboxes,
            predicted_class_confidences,
            targets,
        ) = self.aggregate_prediction_outputs(outputs)

        truth_image_ids = [target["image_id"].detach().item() for target in targets]
        truth_boxes = [
            target["bboxes"].detach()[:, [1, 0, 3, 2]].tolist() for target in targets
        ] # convert to xyxy for evaluation
        truth_labels = [target["labels"].detach().tolist() for target in targets]

        stats = get_coco_stats(
            prediction_image_ids=image_ids,
            predicted_class_confidences=predicted_class_confidences,
            predicted_bboxes=predicted_bboxes,
            predicted_class_labels=predicted_class_labels,
            target_image_ids=truth_image_ids,
            target_bboxes=truth_boxes,
            target_class_labels=truth_labels,
        )['All']
        
        self.log('Average Precision', stats['AP_all'])
        self.log('Average Recall', stats['AR_all'])
        
        return {"val_loss": validation_loss_mean, "metrics": stats}
    
    @typedispatch
    def predict(self, images: List):
        """
        For making predictions from images
        Args:
            images: a list of PIL images

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences
        """
        image_sizes = [(image.size[1], image.size[0]) for image in images]
        images_tensor = torch.stack(
            [
                self.inference_tfms(    # CHECK: class and bboxes is fixed 
                    image=np.array(image, dtype=np.float32),
                    class_labels=np.ones(1),
                    bboxes=np.array([[0, 0, 1, 1]]),
                )["image"]
                for image in images
            ]
        )

        return self._run_inference(images_tensor, image_sizes)
    
    @typedispatch
    def predict(self, images_tensor: torch.Tensor):
        """
        For making predictions from tensors returned from the model's dataloader
        Args:
            images_tensor: the images tensor returned from the dataloader

        Returns: a tuple of lists containing bboxes, predicted_class_labels, predicted_class_confidences
        """
        if images_tensor.ndim == 3:
            images_tensor = images_tensor.unsqueeze(0)
        if images_tensor.shape[-1] != self.img_size or images_tensor.shape[-2] != self.img_size:
            raise ValueError(f"Input tensors must be of shape (N, 3, {self.img_size}, {self.img_size})")

        num_images = images_tensor.shape[0]
        image_sizes = [(self.img_size, self.img_size)] * num_images

        return self._run_inference(images_tensor, image_sizes)
    
    def _run_inference(self, images_tensor, image_sizes):
        # NOTE: the dummies do nothing here, merely exist for developing purposes (avoid to declare
        # another class for models), it has no effect on performance since target only affect loss,
        # which we don't care for this step
        dummy_targets = self._create_dummy_inference_targets(num_images=images_tensor.shape[0])
        detections = self.model(images_tensor.to(self.device), dummy_targets)["detections"]
        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = self.post_process_detections(detections)

        scaled_bboxes = self.__rescale_bboxes(
            predicted_bboxes=predicted_bboxes, image_sizes=image_sizes
        )

        return scaled_bboxes, predicted_class_labels, predicted_class_confidences
    
    def _create_dummy_inference_targets(self, num_images):
        dummy_targets = {
            "bbox": [
                torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device)
                for i in range(num_images)
            ],
            "cls": [torch.tensor([1.0], device=self.device) for i in range(num_images)],
            "img_size": torch.tensor(
                [(self.img_size, self.img_size)] * num_images, device=self.device
            ).float(),
            "img_scale": torch.ones(num_images, device=self.device).float(),
        }

        return dummy_targets
    
    def post_process_detections(self, detections):
        predictions = []
        for i in range(detections.shape[0]):
            predictions.append(
                self._postprocess_single_prediction_detections(detections[i])
            )

        predicted_bboxes, predicted_class_confidences, predicted_class_labels = run_wbf(
            predictions, image_size=self.img_size, iou_thr=self.iou_thres
        )

        return predicted_bboxes, predicted_class_confidences, predicted_class_labels
    
    def _postprocess_single_prediction_detections(self, detections):
        boxes = detections.detach().cpu().numpy()[:, :4]
        scores = detections.detach().cpu().numpy()[:, 4]
        classes = detections.detach().cpu().numpy()[:, 5]
        indexes = np.where(scores > self.predict_confidence_thres)[0]
        boxes = boxes[indexes]

        return {"boxes": boxes, "scores": scores[indexes], "classes": classes[indexes]}
    
    def __rescale_bboxes(self, predicted_bboxes, image_sizes):
        scaled_bboxes = []
        for bboxes, img_dims in zip(predicted_bboxes, image_sizes):
            im_h, im_w = img_dims

            if len(bboxes) > 0:
                scaled_bboxes.append(
                    (np.array(bboxes) * [im_w / self.img_size, im_h / self.img_size,
                                        im_w / self.img_size, im_h / self.img_size]).tolist())
            else:
                scaled_bboxes.append(bboxes)

        return scaled_bboxes
    
    # @patch
    def aggregate_prediction_outputs(self, outputs):
        detections = torch.cat(
            [output["batch_predictions"]["predictions"] for output in outputs]
        )
        image_ids = []
        targets = []
        for output in outputs:
            batch_predictions = output["batch_predictions"]
            image_ids.extend(batch_predictions["image_ids"])
            targets.extend(batch_predictions["targets"])
        (
            predicted_bboxes,
            predicted_class_confidences,
            predicted_class_labels,
        ) = self.post_process_detections(detections)

        return (
            predicted_class_labels,
            image_ids,
            predicted_bboxes,
            predicted_class_confidences,
            targets,
        )
    

if __name__ == '__main__':
    print(type(DataAdapter))