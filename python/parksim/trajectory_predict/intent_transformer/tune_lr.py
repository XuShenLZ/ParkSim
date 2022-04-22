import torch
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from parksim.trajectory_predict.intent_transformer.dataset import IntentTransformerDataModule, IntentTransformerV2DataModule
from parksim.trajectory_predict.intent_transformer.model_utils import split_dataset
from parksim.trajectory_predict.intent_transformer.networks.TrajectoryPredictorWithDecoderIntentCrossAttention import TrajectoryPredictorWithDecoderIntentCrossAttention
from parksim.trajectory_predict.intent_transformer.networks.TrajectoryPredictorVisionTransformer import TrajectoryPredictorVisionTransformer
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
EMPTY_CONFIG = {}
config={
            'dim_model' : 52,
            'num_heads' : 4,
            'dropout' : 0.1426,
            'num_encoder_layers' : 16,
            'num_decoder_layers' : 8,
            'd_hidden' : 256,
            'num_conv_layers' : 2,
    }

if __name__ == '__main__':
    callbacks =  [ModelCheckpoint(dirpath="checkpoints/lr/", monitor=None, mode='min', every_n_train_steps=0, every_n_epochs=1, train_time_interval=None, save_on_train_epoch_end=None)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    trainer = pl.Trainer(accelerator='gpu', devices=1, callbacks=callbacks, default_root_dir="checkpoints/lr/")
    #model = TrajectoryPredictorWithDecoderIntentCrossAttention(config)
    model = TrajectoryPredictorWithDecoderIntentCrossAttention(EMPTY_CONFIG)
    #model = TrajectoryPredictorVisionTransformer(EMPTY_CONFIG)
    custom_dataset_nums = ["../data/DJI_" + str(i).zfill(4) for i in range(7, 26)]
    dataset = IntentTransformerDataModule()
    lr_finder = trainer.tuner.lr_find(model, datamodule=dataset)
    fig = lr_finder.plot(suggest=True)
    plt.show()
    new_lr = lr_finder.suggestion()
    print(new_lr)