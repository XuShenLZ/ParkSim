# ParkSim
![](https://img.shields.io/badge/language-python-blue)
![](https://img.shields.io/github/license/XuShenLZ/ParkSim)
![](https://img.shields.io/badge/ROS-foxy-red)

Vehicle simualtion and behavior prediction in parking lots.

Authors: Xu Shen (xu_shen@berkeley.edu), Alex Wong, Neelay Velingker, Matthew Lacayo, Nidhir Guggilla

## Install
1. Clone this repo
2. In the root folder of this repo, do `pip install -e .`
3. If components of [DLP dataset](https://github.com/MPC-Berkeley/dlp-dataset) is needed, install the DLP package and request data according to the instructions there.
4. If [pytorch](https://pytorch.org/) is needed, please install the correct version based on your OS and hardware.
5. We use [ROS2 Foxy](https://docs.ros.org/en/foxy/index.html) for simulation. You would need that installed if you want to run vehicle simulation.

## [ParkPredict+](https://arxiv.org/abs/2204.10777)

ParkPredict+: Multimodal Intent and Motion Prediction for Vehicles in Parking Lots with CNN and Transformer

Authors: Xu Shen, Matthew Lacayo, Nidhir Guggilla, Francesco Borrelli

> **Note**: You don't need ROS for ParkPredict+.

### Intent Prediction
1. `python/parksim/intent_predict/cnnV2` contains code for building, training, and evaluating the model.
2. A pre-trained model can be [downloaded here](https://drive.google.com/file/d/1LVQJRQmjGfGchxhMRchiZRCjrlFDVch-/view?usp=sharing).
3. Use `python/parksim/intent_predict/cnnV2/data_processing/create_dataset.py` to generate training data from DLP dataset.

### Trajectory Prediction
1. `python/parksim/trajectory_predict/intent_transformer` contains code for building, training, and evaluating the model.
2. `python/parksim/trajectory_predict/intent_transformer/networks/TrajectoryPredictorWithDecoderIntentCrossAttention.py` is the trajectory prediction model presented in the paper.
3. `python/parksim/trajectory_predict/intent_transformer/testing_multimodal.ipynb` demonstrates the multimodal preidction.
4. A pre-trained model can be [downloaded here](https://drive.google.com/file/d/1c9KQXwFMRIYPJo1sXJKepoBcrEme_HxU/view?usp=sharing).
5. For training the network, use `python/parksim/trajectory_predict/data_processing/create_dataset.py` to generate features and labels based on DLP dataset. Then use `python/parksim/trajectory_predict/data_processing/prepare_single_file_data.py` to further process the generated files. Then you can use `python/parksim/trajectory_predict/intent_transformer/train.py` to train the network.
