import torch
import os
from parksim.trajectory_predict.intent_transformer.dataset import IntentTransformerV2DataModule
from parksim.trajectory_predict.intent_transformer.networks.TrajectoryPredictorVisionTransformer import TrajectoryPredictorVisionTransformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

MODEL_LABEL = 'VisionTransformer'
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    config={
            'dim_model' : 64,
            'num_heads' : 8,
            'dropout' : 0.10,
            'num_encoder_layers' : 2,
            'num_decoder_layers' : 2,
            'd_hidden' : 256,
            'patch_size' : 20,
            'loss' : 'L1'
    }

    custom_dataset_nums = ["../data/DJI_" + str(i).zfill(4) for i in range(7, 13)]
    datamodule = IntentTransformerV2DataModule(all_dataset_nums=custom_dataset_nums)
    model = TrajectoryPredictorVisionTransformer(config)
    patience = 25
    earlystopping = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=3,
        mode="min",
        every_n_epochs=1
    )

    callbacks=[earlystopping, checkpoint_callback]
    trainer = pl.Trainer(accelerator="gpu", devices=1, default_root_dir=f"checkpoints/{MODEL_LABEL}/", callbacks=callbacks, track_grad_norm=2)
    trainer.fit(model=model, datamodule=datamodule)

