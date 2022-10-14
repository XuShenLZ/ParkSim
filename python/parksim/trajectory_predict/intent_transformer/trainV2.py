import torch
import os
from parksim.trajectory_predict.intent_transformer.dataset import IntentTransformerV2DataModule
from parksim.trajectory_predict.intent_transformer.models.trajectory_predictor_vision_transformer import TrajectoryPredictorVisionTransformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

MODEL_LABEL = 'VisionTransformer'
DEFAULT_CONFIG = {}
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    """
    SPECIFY CONFIG HERE:
    """
    config={
            'dim_model' : 512,
            'num_heads' : 16,
            'dropout' : 0.10,
            'num_encoder_layers' : 2,
            'num_decoder_layers' : 4,
            'd_hidden' : 256,
            'patch_size' : 20,
            'loss' : 'L1'
    }

    """
    SPECIFY MODEL HERE:
    """
    model = TrajectoryPredictorVisionTransformer(config)
    #model = TrajectoryPredictorVisionTransformer(DEFAULT_CONFIG)


    datamodule = IntentTransformerV2DataModule()
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
    trainer = pl.Trainer(accelerator="gpu", devices=1, default_root_dir=f"checkpoints/{MODEL_LABEL}/", callbacks=callbacks, track_grad_norm=2)
    trainer.fit(model=model, datamodule=datamodule)

