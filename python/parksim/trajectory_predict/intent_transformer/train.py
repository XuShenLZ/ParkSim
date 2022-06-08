import torch
import os
from parksim.trajectory_predict.intent_transformer.dataset import IntentTransformerDataModule
from parksim.trajectory_predict.intent_transformer.networks.TrajectoryPredictorWithDecoderIntentCrossAttention import TrajectoryPredictorWithDecoderIntentCrossAttention
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

MODEL_LABEL = 'TrajectoryPredictorWithDecoderIntentCrossAttention'
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    config={
        'dim_model': 64, 
        'num_heads': 16, 
        'dropout': 0.22064449902863661, 
        'num_encoder_layers': 2, 
        'num_decoder_layers': 4, 
        'd_hidden': 128, 
        'detach_cnn': False
    }

    model = TrajectoryPredictorWithDecoderIntentCrossAttention(config)
    datamodule = IntentTransformerDataModule()
    patience = 25
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