from torchmetrics import MeanSquaredError
import pytorch_lightning as pl

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.nn as nn
import torch
import utils # needed for masking

from sklearn.preprocessing import minmax_scale


class SeqNet(pl.LightningModule):
    def __init__(self, backbone_net=None, n_epochs=None, forecast_window=None, enc_seq_len=None):
        super().__init__()

        if backbone_net is None:
            raise Exception('backbone_net missing in arguments list')
                
        self.model = backbone_net 
        
        self.n_epochs = n_epochs
        
        self.tgt_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_window,
            dim2=forecast_window
        )

        self.src_mask = utils.generate_square_subsequent_mask(
            dim1=forecast_window,
            dim2=enc_seq_len
        )
        
        self.criterion = nn.MSELoss()
        
        self.mae = nn.L1Loss()
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()

    def forward(self, x):
        
        src, tgt, _ = x
        
        src_mask, tgt_mask = self.src_mask.to(self.device), self.tgt_mask.to(self.device)
        
        return self.model(src, tgt, src_mask, tgt_mask)
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_epochs // 3, eta_min=1e-5)
        return [optimizer], [scheduler]
    
    def training_step(self, train_batch, batch_idx):
        _, _, tgt_y = train_batch

        prediction = self.forward(train_batch).squeeze()
        
        loss = self.criterion(tgt_y.squeeze(), prediction)
        
        self.train_mse(tgt_y.squeeze(), prediction)
        self.log('train_MSE', self.train_mse, on_step=True, on_epoch=False)
        self.log('train_RMSE', torch.sqrt(loss))
        self.log('train_MAE', self.mae(tgt_y.squeeze(), prediction))
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        _, _, tgt_y = val_batch

        prediction = self.forward(val_batch).squeeze()
        
        loss = self.criterion(tgt_y.squeeze(), prediction)
        
        self.val_mse(tgt_y.squeeze(), prediction)
        self.log('val_MSE', self.val_mse, on_step=False, on_epoch=True)
        self.log('val_RMSE', torch.sqrt(loss))
        self.log('val_MAE', self.mae(tgt_y.squeeze(), prediction))
        
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        prediction = self.forward(batch)
        
        return prediction