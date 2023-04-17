import torch
import torch.nn.functional as F
import pytorch_lightning as pl

class BaseMGCVAELightningModule(pl.LightningModule):
    # TODO: fill this in
    # TODO: fill in the correct loss function
    def __init__(self, config: dict, input_shape=(), loss_fn=F.l1_loss) -> None:
        super().__init__()
        self.save_hyperparameters()
        # TODO: use the correct lr
        self.lr = 1e-3
        self.loss_fn = loss_fn
    
    def configure_optimizers(self):
        # TODO: do I need to change this?
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_total_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }
        return [optimizer], [lr_scheduler_config]
    
    # TODO: fill in the loss fnct from paper
    def get_loss(self, pred, label, loss_type):
        total_loss = self.loss_fn(pred, label)
        with torch.no_grad():
            metrics = {
                f"{loss_type}_total_loss" : total_loss
            }
            self.log_dict(metrics)
        return total_loss, metrics
    
    # TODO: fill in training step from paper
    def training_step(self, batch, _batch_idx):
        ego_history, target_history, neighbor_veh_history, neighbor_ped_history, target_future, semantic_map = batch
        pred = self(ego_history, target_history, neighbor_veh_history, neighbor_ped_history, semantic_map)
        loss, _metrics = self.get_loss(pred, target_future, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        ego_history, target_history, neighbor_veh_history, neighbor_ped_history, target_future, semantic_map = batch
        pred = self(ego_history, target_history, neighbor_veh_history, neighbor_ped_history, semantic_map)
        _loss, metrics = self.get_loss(pred, target_future, "val")
        return metrics
    
    def test_step(self, batch, batch_idx):
        ego_history, target_history, neighbor_veh_history, neighbor_ped_history, target_future, semantic_map = batch
        pred = self(ego_history, target_history, neighbor_veh_history, neighbor_ped_history, semantic_map)
        _loss, metrics = self.get_loss(pred, target_future, "test")
        return metrics
