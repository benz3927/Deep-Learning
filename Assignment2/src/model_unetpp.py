# src/model_unetpp.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import segmentation_models_pytorch as smp

class UNetPlusPlusModule(pl.LightningModule):
    def __init__(self, classes=3, lr=1e-3, weight_decay=1e-4, encoder_name="resnet34"):
        super().__init__()
        self.save_hyperparameters()
        
        # Use smp UNet++ with configurable encoder
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=classes,
            activation=None  # We'll use softmax/argmax manually
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        self.classes = classes

        # Metrics with proper task specification
        self.train_iou = torchmetrics.JaccardIndex(
            num_classes=classes,
            task="multiclass"
        )
        self.val_iou = torchmetrics.JaccardIndex(
            num_classes=classes,
            task="multiclass"
        )
        self.val_dice = torchmetrics.F1Score(
        num_classes=classes,
        task="multiclass",
        average="macro"
        )

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        imgs, masks = batch
        logits = self(imgs)
        loss = self.loss_fn(logits, masks)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, masks

    def training_step(self, batch, batch_idx):
        loss, preds, masks = self.step(batch)
        
        # Log loss per step, metrics per epoch
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        
        # Update metric (accumulated internally)
        self.train_iou(preds, masks)
        self.log("train_mIoU", self.train_iou, on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, masks = self.step(batch)
        
        # Log validation metrics
        self.log("val_loss", loss, prog_bar=True)
        
        # Update metrics
        self.val_iou(preds, masks)
        self.val_dice(preds, masks)
        
        self.log("val_mIoU", self.val_iou, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_Dice", self.val_dice, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Use AdamW for weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Add learning rate scheduler (common in segmentation tasks)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }