import torch
import torch.nn.functional as F

import torchmetrics
import torchmetrics.functional as TMF

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class ModelInterface(pl.LightningModule):
    def __init__(self, model, learning_rate, len_train_dataloader):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.len_train_dataloader = len_train_dataloader

        self.validation_step_outputs = []
        self.validation_step_targets = []
        
    def step(self, batch):
        X, y = batch
        preds = self.model(X)
        loss = F.cross_entropy(preds, y)
        
        return preds, y, loss

    def training_step(self, batch, batch_idx):
        _, _, loss = self.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def compute_metrics(self, preds, targets):
        auroc_ = TMF.auroc(preds, targets, task="multiclass", num_classes=2)
        auprc_ = TMF.average_precision(preds, targets, task="multiclass", num_classes=2)

        sensitivity_ = TMF.recall(preds, targets, task="multiclass", num_classes=2)
        specificity_ = TMF.specificity(preds, targets, task="multiclass", num_classes=2)
        f1_score_ = TMF.f1_score(preds, targets, task="multiclass", num_classes=2)
        # try:
        #     auroc_ = TMF.auroc(preds, targets, task="multiclass", num_classes=2)
        #     auprc_ = TMF.average_precision(preds, targets, task="multiclass", num_classes=2)

        #     sensitivity_ = TMF.recall(preds, targets, task="multiclass", num_classes=2)
        #     specificity_ = TMF.specificity(preds, targets, task="multiclass", num_classes=2)
        #     f1_score_ = TMF.f1_score(preds, targets, task="multiclass", num_classes=2)
        # except:
        #     auroc_ = TMF.auroc(preds, targets, num_classes=2)
        #     auprc_ = TMF.average_precision(preds, targets, num_classes=2)

        #     sensitivity_ = TMF.recall(preds, targets, num_classes=2)
        #     specificity_ = TMF.specificity(preds, targets, num_classes=2)
        #     f1_score_ = TMF.f1_score(preds, targets, num_classes=2)

        return auroc_, auprc_, sensitivity_, specificity_, f1_score_
    
    def logging(self, logging_objects, mode="valid"):
        auroc_, auprc_, sensitivity_, specificity_, f1_score_ = logging_objects
        
        self.log(f"{mode}_auroc", auroc_, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_auprc", auprc_, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_f1", f1_score_, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_sensitivity", sensitivity_, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{mode}_specificity", specificity_, on_step=False, on_epoch=True, prog_bar=True)
       
    def validation_step(self, batch, batch_idx):
        preds, y, loss = self.step(batch)
        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.validation_step_outputs.append(preds)
        self.validation_step_targets.append(y)

    def on_validation_epoch_end(self):
        preds = torch.cat([tmp for tmp in self.validation_step_outputs])
        targets = torch.cat([tmp for tmp in self.validation_step_targets])
        
        logging_objects = self.compute_metrics(preds, targets)
        self.logging(logging_objects)
        
        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()
    
    def test_step(self, batch, batch_idx):
        preds, y, loss = self.step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.validation_step_outputs.append(preds)
        self.validation_step_targets.append(y)
    
    def on_test_epoch_end(self):
        preds = torch.cat([tmp for tmp in self.validation_step_outputs])
        targets = torch.cat([tmp for tmp in self.validation_step_targets])
        
        logging_objects = self.compute_metrics(preds, targets)
        self.logging(logging_objects, mode="test")
        
        try:
            conf_mat = TMF.confusion_matrix(preds, targets, task="multiclass", num_classes=2)
        except:
            conf_mat = TMF.confusion_matrix(preds, targets, num_classes=2)
        print(conf_mat.detach().cpu().numpy())    
        
        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()
    
    def predict_step(self, batch, batch_idx):
        X, y = batch
        feats = self.model.forward_features(X)
        feats = torch.mean(torch.mean(feats, dim=-1), dim=-1)
        
        return feats, y
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10 * self.len_train_dataloader)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
          
def define_callback(PROJECT_NAME):
    return [ModelCheckpoint(monitor='valid_loss', mode="min", save_top_k=1, dirpath=f'weights/{PROJECT_NAME}', 
                            filename='{epoch:03d}-{valid_loss:.4f}-{valid_auroc:.4f}-{valid_auprc:.4f}')]
            

# predictor = ModelInterface(model)
# trainer = pl.Trainer(max_epochs=30, gpus=1, enable_progress_bar=True, 
#                      callbacks=callbacks, precision=16)