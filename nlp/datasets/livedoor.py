import os


class Livedoor(object):

    exclude_files = [
        'CHANGES.txt',
        'README.txt',
        'LICENSE.txt',
    ]

    def __init__(self, path):
        self.base_path = path
        self.label2id = self.__label2id()
        self.texts = []
        self.labels = []

    def __label2id(self):
        label2id = {}
        dirs = [e for e in os.listdir(self.base_path)
                if os.path.isdir(os.path.join(self.base_path, e))]
        for root, _, _ in os.walk(self.base_path):
            parent_dir = os.path.basename(root)
            if parent_dir in dirs and parent_dir not in label2id:
                label2id[parent_dir] = len(label2id)
        return label2id

    def get_data(self):
        if not self.texts and not self.labels:
            for root, _, files in os.walk(self.base_path):
                parent_dir = os.path.basename(root)
                for file in files:
                    if file in self.exclude_files:
                        continue
                    text = ''
                    for index, line in enumerate(open(os.path.join(root, file), 'r')):
                        if index <= 1:  # Ignore top 2 lines
                            continue
                        text += line
                    self.texts.append(text)
                    self.labels.append(self.label2id[parent_dir])
        return self.texts, self.labels
