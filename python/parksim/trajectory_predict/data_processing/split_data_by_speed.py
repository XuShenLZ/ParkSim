import numpy as np
import os

_CURRENT = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(_CURRENT, '..', 'data')

THRESHOLD = 1.5
name = 'DJI_0022'

image_history = np.load(os.path.join(DATA_PATH, '%s_image_history.npy' % name))

trajectory_history = np.load(os.path.join(DATA_PATH,'%s_trajectory_history.npy' % name))

trajectory_future = np.load(os.path.join(DATA_PATH,'%s_trajectory_future.npy' % name))
    
intent_pose = np.load(os.path.join(DATA_PATH, '%s_intent_pose.npy' % name))

slow_indices = []
fast_indices = []
for i in range(image_history.shape[0]):
    avg_speed = np.linalg.norm(trajectory_history[i, 0, :2] - trajectory_history[i, -1, :2]) / 4
    if avg_speed > THRESHOLD:
        fast_indices.append(i)
    else:
        slow_indices.append(i)

np.save(os.path.join(DATA_PATH, '%sslow_image_history.npy' % name), np.take(image_history, slow_indices, axis=0))
np.save(os.path.join(DATA_PATH,'%sslow_trajectory_history.npy' % name), np.take(trajectory_history, slow_indices, axis=0))
np.save(os.path.join(DATA_PATH, '%sslow_trajectory_future.npy' % name), np.take(trajectory_future, slow_indices, axis=0))
np.save(os.path.join(DATA_PATH, '%sslow_intent_pose.npy' % name), np.take(intent_pose, slow_indices, axis=0))

np.save(os.path.join(DATA_PATH, '%sfast_image_history.npy' % name), np.take(image_history, fast_indices, axis=0))
np.save(os.path.join(DATA_PATH,'%sfast_trajectory_history.npy' % name), np.take(trajectory_history, fast_indices, axis=0))
np.save(os.path.join(DATA_PATH, '%sfast_trajectory_future.npy' % name), np.take(trajectory_future, fast_indices, axis=0))
np.save(os.path.join(DATA_PATH, '%sfast_intent_pose.npy' % name), np.take(intent_pose, fast_indices, axis=0))
