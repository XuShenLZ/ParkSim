# Models and Data Modules
There are two types of models: V1 and V2 models. V1 models have an image for
each step of the trajectory history, while V2 models only use the final image
in the trajectory history. In practice, this drastically reduces the memory
usage of V2 models compared to V1. Moreover, we have found the performance of V2
models to be on par with V1 models, so we recommend using V2 models exclusively.
In the future, all models will be rewritten to support a V1 dataset and a V2
dataset, but currently the breakdown is as follows:

V1 Models: TrajectoryPredictorWithDecoderIntentCrossAttention

V2 Models: All others

# Generating Training Data
To generate the training data for yourself, please inspect
trajectory_predict/data_processing/prepare_single_file_data.py

Inside of this file, please specify the DJI files you would like to generate
from. Currently, it is set to range(1, 31), which corresponds to DJI files 1
through 30. However, for testing purposes this can be changed to something much
more reasonable, like [12, 13]. 

To actually run the file, simply run 

python trajectory_predict/data_processing/prepare_single_file_data.py

# Training Models
If you wish to train a V1 model, please use trainV1.py inside
trajectory_predict/intent_transformer, otherwise use trainV2.py

To actually train, please specify a config inside of the training file. An
example config is as follows:

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

Then, you can specify your model with a statement like:

model = \<Model you want to use here\>(config)

Finally, to run the training, simply run:
 
python trajectory_predict/intent_transformer/trainV\<Version Number Here\>.py

