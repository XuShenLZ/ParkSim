from functools import partial
from ray import tune
import os
from parksim.trajectory_predict.intent_transformer.dataset import IntentTransformerDataModule, IntentTransformerV2DataModule
from parksim.trajectory_predict.intent_transformer.networks.TrajectoryPredictorVisionTransformer import TrajectoryPredictorVisionTransformer
from parksim.trajectory_predict.intent_transformer.networks.TrajectoryPredictorWithEncoderImageCrossAttention import TrajectoryPredictorWithEncoderImageCrossAttention
import pytorch_lightning as pl
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

def train_model(config, model_class):
    # Create your PTL model.
    model = model_class(config)
    datamodule = IntentTransformerV2DataModule(batch_size=512, num_workers=6)

    # Create the Tune Reporting Callback
    metrics = {"train_loss": "train_total_loss", "val_loss": "val_total_loss"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]
    trainer = pl.Trainer(accelerator="gpu", devices=1, default_root_dir=f"checkpoints/hp_tuning/{model.__class__.__name__}/")
    trainer.tune(model, datamodule=datamodule)
    optimal_lr = model.lr
    print(f"Optimal LR: {optimal_lr}")
    model = model_class(config)
    trainer = pl.Trainer(accelerator="gpu", devices=1, default_root_dir=f"checkpoints/hp_tuning/{model.__class__.__name__}/", max_epochs=5, callbacks=callbacks)
    model.lr = optimal_lr
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    """
    If you are getting errors relating to 'paging file', please check the 
    defined datamodule in the train_model function above. Try reducing the batch
    size, or decreasimg num_workers for the datamodule.
    """
    config={
        'dim_model' : tune.choice([64, 128, 256]),
        'num_heads' : tune.choice([8, 16, 32]),
        'dropout' : tune.uniform(0.01, 0.4),
        'num_encoder_layers' : tune.choice([2, 4, 6, 8]),
        'num_decoder_layers' : tune.choice([2,4,6,8]),
        'd_hidden' : tune.choice([128, 256, 512]),
        'num_conv_layers' : tune.choice([2, 4, 8]),
        'num_cnn_features' : tune.choice([64, 128, 256, 1024]),
    }

    model_class = TrajectoryPredictorWithEncoderImageCrossAttention
    num_samples = 20
    scheduler = ASHAScheduler(
        max_t=3,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["dim_model", "num_img_encoder_layers", "num_trajectory_encoder_layers", "num_decoder_layers"],
        metric_columns=["val_loss", "training_iteration"])
    resources_per_trial = {"cpu": 8, "gpu": 1}
    analysis = tune.run(
        partial(train_model, model_class=model_class),
        resources_per_trial=resources_per_trial,
        metric="val_loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        max_concurrent_trials=1,
        progress_reporter=reporter)
    print("Best hyperparameters found were: ", analysis.best_config)

