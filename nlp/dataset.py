import os


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
