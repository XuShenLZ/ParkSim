import numpy as np
from tqdm import tqdm


def create_memmap(file_paths, train_output_path, val_output_path, test_output_path, proportions, extra_shape, shufflers):
    curr_train_idx = 0
    curr_val_idx = 0
    curr_test_idx = 0
    print(extra_shape)
    for path, shuffler in tqdm(list(zip(file_paths, shufflers))):
        curr_data = np.load(path, mmap_mode='c')[shuffler]
        curr_points = curr_data.shape[0]
        train_idx = int(proportions[0] * curr_points)
        val_idx = train_idx + int(proportions[1] * curr_points)
        test_idx = curr_points
        train_memmap = np.memmap(train_output_path, dtype='float32', mode='r+', shape=(train_idx,*extra_shape), offset=np.longlong(curr_train_idx) * np.longlong(4) * np.longlong(np.prod(extra_shape)))
        val_memmap = np.memmap(val_output_path, dtype='float32', mode='r+', shape=(val_idx - train_idx,*extra_shape), offset=np.longlong(curr_val_idx) * np.longlong(4) * np.longlong(np.prod(extra_shape)))
        test_memmap = np.memmap(test_output_path, dtype='float32', mode='r+', shape=(test_idx - val_idx,*extra_shape), offset=np.longlong(curr_test_idx) * np.longlong(4) * np.longlong(np.prod(extra_shape)))
        train_memmap[:] = curr_data[:train_idx]
        val_memmap[:] = curr_data[train_idx:val_idx]
        test_memmap[:] = curr_data[val_idx:]
        curr_train_idx += train_idx
        curr_val_idx += (val_idx - train_idx)
        curr_test_idx += (test_idx - val_idx)
        train_memmap.flush()
        val_memmap.flush()
        test_memmap.flush()


def create_all_memmaps(dataset_nums, proportions, num_train, num_val, num_test):
    intent_file_paths = [f'{file_path}_intent_pose.npy' for file_path in dataset_nums]
    shufflers = [np.random.permutation(np.load(path).shape[0]) for path in intent_file_paths]

    image_history_file_paths = [f'{file_path}_image_history.npy' for file_path in dataset_nums]
    train_image_output_path = f"../data/train_image_history.npy"
    val_image_output_path = f"../data/val_image_history.npy"
    test_image_output_path = f"../data/test_image_history.npy"
    first_memmap = np.load(image_history_file_paths[0], mmap_mode='c')
    extra_shape = first_memmap.shape[1:]
    train_memmap = np.memmap(train_image_output_path, dtype='float32', mode='w+', shape=(num_train,*extra_shape))
    val_memmap = np.memmap(val_image_output_path, dtype='float32', mode='w+', shape=(num_val,*extra_shape))
    test_memmap = np.memmap(test_image_output_path, dtype='float32', mode='w+', shape=(num_test,*extra_shape))
    create_memmap(image_history_file_paths, train_image_output_path, val_image_output_path, test_image_output_path, proportions, extra_shape, shufflers)
    

    trajectory_history_file_paths = [f'{file_path}_trajectory_history.npy' for file_path in dataset_nums]
    train_trajectory_history_output_path = f"../data/train_trajectory_history.npy"
    val_trajectory_history_output_path = f"../data/val_trajectory_history.npy"
    test_trajectory_history_output_path = f"../data/test_trajectory_history.npy"
    first_memmap = np.load(trajectory_history_file_paths[0], mmap_mode='c')
    extra_shape = first_memmap.shape[1:]
    train_memmap = np.memmap(train_trajectory_history_output_path, dtype='float32', mode='w+', shape=(num_train,*extra_shape))
    val_memmap = np.memmap(val_trajectory_history_output_path, dtype='float32', mode='w+', shape=(num_val,*extra_shape))
    test_memmap = np.memmap(test_trajectory_history_output_path, dtype='float32', mode='w+', shape=(num_test,*extra_shape))
    create_memmap(trajectory_history_file_paths, train_trajectory_history_output_path, val_trajectory_history_output_path, test_trajectory_history_output_path, proportions, extra_shape, shufflers)

    intent_file_paths = [f'{file_path}_intent_pose.npy' for file_path in dataset_nums]
    train_intent_output_path = f"../data/train_intent.npy"
    val_intent_output_path = f"../data/val_intent.npy"
    test_intent_output_path = f"../data/test_intent.npy"
    first_memmap = np.load(intent_file_paths[0], mmap_mode='c')
    extra_shape = first_memmap.shape[1:]
    train_memmap = np.memmap(train_intent_output_path, dtype='float32', mode='w+', shape=(num_train,*extra_shape))
    val_memmap = np.memmap(val_intent_output_path, dtype='float32', mode='w+', shape=(num_val,*extra_shape))
    test_memmap = np.memmap(test_intent_output_path, dtype='float32', mode='w+', shape=(num_test,*extra_shape))
    create_memmap(intent_file_paths, train_intent_output_path, val_intent_output_path, test_intent_output_path, proportions, extra_shape, shufflers)

    trajectory_future_file_paths = [f'{file_path}_trajectory_future.npy' for file_path in dataset_nums]
    train_trajectory_future_output_path = f"../data/train_trajectory_future.npy"
    val_trajectory_future_output_path = f"../data/val_trajectory_future.npy"
    test_trajectory_future_output_path = f"../data/test_trajectory_future.npy"
    first_memmap = np.load(trajectory_future_file_paths[0], mmap_mode='c')
    extra_shape = first_memmap.shape[1:]
    train_memmap = np.memmap(train_trajectory_future_output_path, dtype='float32', mode='w+', shape=(num_train,*extra_shape))
    val_memmap = np.memmap(val_trajectory_future_output_path, dtype='float32', mode='w+', shape=(num_val,*extra_shape))
    test_memmap = np.memmap(test_trajectory_future_output_path, dtype='float32', mode='w+', shape=(num_test,*extra_shape))
    create_memmap(trajectory_future_file_paths, train_trajectory_future_output_path, val_trajectory_future_output_path, test_trajectory_future_output_path, proportions, extra_shape, shufflers)


if __name__ == '__main__':
    datasets_to_generate = range(1, 31)
    custom_dataset_nums = ["../data/DJI_" + str(i).zfill(4) for i in datasets_to_generate]
    proportions = [0.85, 0.1, 0.05]
    train_size = 0
    val_size = 0
    test_size = 0

    intent_file_paths = [f'{file_path}_intent_pose.npy' for file_path in custom_dataset_nums]
    for fp in intent_file_paths:
        sample_data = np.load(fp)
        num_points = sample_data.shape[0]
        train_size += int(proportions[0] * num_points)
        val_size += int(proportions[1] * num_points)
        test_size += (num_points - int(proportions[0] * num_points) - int(proportions[1] * num_points))

    print(train_size, val_size, test_size)
    np.save('../data/dataset_sizes.npy', [train_size, val_size, test_size])
    create_all_memmaps(custom_dataset_nums, proportions, train_size, val_size, test_size)

  

    

