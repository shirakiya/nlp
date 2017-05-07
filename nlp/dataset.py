import os
import numpy as np


def file_generator(path):
    if isinstance(path, str):
        path = list(path)
    for file_path in path:
        with open(file_path, 'r') as f:
            while True:
                text = f.readline()
                if text == '':
                    break
                yield text


def get_files_list(path, exclude_files=[]):
    filelist = []
    if os.path.isfile(path):
        filelist.append(path)
    else:
        for root, _, files in os.walk(path):
            filelist.extend([os.path.join(root, f) for f in files if f not in exclude_files])
    return filelist


def generate_batch(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
