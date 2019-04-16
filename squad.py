""" Squad Dataset Downloader, parser and prepares data to be fed into Neural Network"""

import json
import os
import urllib.request

# import torchtext.vocab as vocab

__all__ = ["Squad"]


class Squad():
    def __init__(self, train=False, shuffle=True, download=False, batch_size=32, root="data"):
        """        
        :param train: True for training data, False for test data
        :param shuffle: Shuffle data
        :param download: True / False. Download files or not?
        :param root: the dir to save the downloaded train / test files
        """
        # Devise ways to download it manually
        self.train = train
        self.root = root
        self.batch_size = batch_size

        self.vectors = []  # Word embeddings
        self.vocab = []  # Vocabulary of the dataset
        self.specials = ['<sos>', '<pad>', '<eos>']

        train_filename = "train-v2.0.json"
        test_filename = "dev-v2.0.json",
        self.filename = train_filename if train else test_filename
        self.filename = os.path.join(root, self.filename)

        url_type = "train" if train else "test"
        urls = {
            "test": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
            "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
        }
        self.url = urls[url_type]

        if download:
            self.download()

        if not os.path.exists(self.filename):
            raise RuntimeError("{} Dataset not found. You can use download=True to download it".format(self.filename))

        self.process()

    def process(self):
        """ Process the Squad file """
        self.data = []

        # Read json file and do stuff
        with open(self.filename) as json_file:
            sets = json.load(json_file)['data']
            print("Number of paragraphs / sets :", len(sets))
            for i, set in enumerate(sets):
                paragraphs = set['paragraphs']
                for paragraph in paragraphs:
                    qas = paragraph['qas']
                    context = paragraph['context'].lower()
                    qas_ = []
                    for qa in qas:
                        question = qa['question']
                        is_impossible = qa['is_impossible']
                        if not is_impossible:
                            answer = qa['answers'][0]['text']
                            answer_start = qa['answers'][0]['answer_start']
                        else:
                            answer = ""
                            answer_start = -1
                        qas_.append((question, answer, answer_start, is_impossible))
                    self.data.append((context, qas_))
        self.data = iter(self.data)

    def download(self):
        """ Download the request file and save it """
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print("Downloading ... ", self.filename)
        urllib.request.urlretrieve(self.url, self.filename)

    def __iter__(self):
        """
        Returns
        context, qas
        qas = (question, answer, answer_start, is_impossible)
        """
        return self

    def __next__(self):
        """
        Returns
        context, qas
        qas = (question, answer, answer_start, is_impossible)
        """
        return next(self.data)

    def next(self):
        return self.__next__()


if __name__ == '__main__':
    s = Squad(train=True, root="data", download=False)
    for i, data in enumerate(s):
        context, qas = data
        print(i, "QA pairs : ", len(qas))



# vocab = vocab.Vocab(osp.words, 2000, 3, specials=['<eos>', '<sos>', '<pad>'], vectors="glove.6B.100d",
#                     vectors_cache='../.vector_cache')
# print(vocab.vectors[4])
