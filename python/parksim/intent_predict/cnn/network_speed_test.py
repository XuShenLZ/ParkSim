from parksim.intent_predict.cnn.models.small_regularized_cnn import SmallRegularizedCNN
import time
from parksim.intent_predict.cnn.models import Predictor
from dlp.dataset import Dataset
from parksim.intent_predict.cnn.data_processing.utils import CNNDataProcessor
from pathlib import Path


model = SmallRegularizedCNN().eval()



# Load dataset
ds = Dataset()

home_path = str(Path.home() / 'Documents/Github' / 'dlp-dataset/data/DJI_0012')
ds.load(home_path)
scene = ds.get('scene', ds.list_scenes()[0])
frame_index = 20
frame = ds.get_future_frames(scene['first_frame'],timesteps=300)[frame_index]
all_instance_tokens = frame['instances']

extractor = CNNDataProcessor(ds = ds)
inst_token = frame['instances'][1]
instance = ds.get('instance', inst_token)
img_frame = extractor.vis.plot_frame(frame['frame_token'])
instance_start_time = time.time()
img = extractor.vis.inst_centric(img_frame, inst_token)
instance_end_time = time.time()

global_coords = instance['coords']
heading = instance['heading']
speed = instance['speed']
time_in_lot = 0.04 / 60 * frame_index

predictor = Predictor(use_cuda=False)
predictor.load_model(extractor.waypoints_graph)
predictor.model = model
start_time = time.time()
response = predictor.predict(img, global_coords, heading, speed, time_in_lot)
end_time = time.time()

print("Instance Time: " + str(instance_end_time - instance_start_time))
print("Pytorch Time: ", end_time - start_time)
print("Num Spots Considered: ", len(response.distribution))