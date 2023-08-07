import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from parksim.trajectronpp.dataset import MGCVAEDataModule
from parksim.trajectronpp.models.mgcvae import TrajectoryPredictorMGCVAE

MODEL_LABEL = 'Trajectron++'
DEFAULT_CONFIG = {}
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    """
    SPECIFY CONFIG HERE:
    """
    # TODO: fill in config
    config={
        'N': 1, # unclear tbh
        'K': 25, # number of samples from z dist
        'dec_rnn_dim': 128, # output dimension for decoder GRU
        'GMM_components': 1, # components in decoder GMM
        "log_p_yt_xz_max": 6, # max log prob in y dist
        "prediction_horizon": 10, # future timesteps to predict
    }

    """
    SPECIFY MODEL HERE:
    """
    model = TrajectoryPredictorMGCVAE(config)


    datamodule = MGCVAEDataModule()
    patience = 10
    earlystopping = EarlyStopping(monitor="val_total_loss", mode="min", patience=patience)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_total_loss",
        filename="{epoch}-{val_total_loss:.4f}",
        save_top_k=3,
        mode="min",
        every_n_epochs=1
    )

    callbacks=[earlystopping, checkpoint_callback]
    # trainer = pl.Trainer(accelerator="gpu", devices=1, default_root_dir=f"checkpoints/{MODEL_LABEL}/", callbacks=callbacks, track_grad_norm=2)
    # TODO: tune these. setting max_epochs to 100 for testing
    trainer = pl.Trainer(accelerator="gpu", devices=1, default_root_dir=f"checkpoints/{MODEL_LABEL}/", callbacks=callbacks, track_grad_norm=2, max_epochs=100)
    trainer.fit(model=model, datamodule=datamodule)
