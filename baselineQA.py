from collections import Counter
import os
import pickle
from bisect import bisect_left

from nltk import WordPunctTokenizer, WordNetLemmatizer, PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import squad
from preprocessing.squad_preprocess import tokenize

tokenizer = WordPunctTokenizer()
lemmatizer = WordNetLemmatizer()
sent_tokenizer = PunktSentenceTokenizer()

UNK = "<unk>"

dir = "baseline"
if not os.path.exists(dir): os.makedirs(dir)
vector_file = os.path.join(dir, "tfidf_vectors")
vocab_file = os.path.join(dir, "vocab_squad")


def in_vocab(lists, item):
    """
    Returns True if the item is in thesorted list else False
    """
    index = bisect_left(lists, item)
    if lists[min(index, len(lists) - 1)] == item:
        return True
    else:
        return False


def compute_vectors():
    """ Computes tfidf vectors for the dataset and pickles the vectorizer """
    files = ["answer", "question", "context"]
    types = ["dev", "train"]
    files = ["{}.{}".format(t, f) for f in files for t in types]
    files = [(os.path.join("data", file)) for file in files]

    text = []
    for file in files:
        text.append(open(file, mode='r', encoding='utf8').read())

    text = " ".join(text)
    sentences = text.splitlines()

    words = tokenize(text)
    vocab = Counter(words).most_common(10000)
    vocab = [word for word, count in vocab]
    vocab.append(UNK)
    vocab = sorted(vocab)
    print("Vocab size : ", len(vocab))
    with open(vocab_file, mode='w', encoding='utf8') as file:
        file.write("\n".join(vocab))

    # Add more text to aid tf-idf computation
    print("Processing sentences") 
    base_text = open("baseline/base", encoding='utf8', mode='r').readlines()
    for sentence in base_text:
        sentence = tokenize(sentence)
        sentence = [word if in_vocab(vocab, word) else UNK for word in sentence]
        sentences.append(" ".join(sentence))

    print("Fitting vectorizer") 
    vectorizer = TfidfVectorizer().fit(sentences)
    pickle.dump(vectorizer, open(vector_file, mode='wb'))


def is_correct(contexts, rank, answer_start, answer_end):
    """
    :return: True if the prediction covers ground truth
    """
    lens = [len(context) for context in contexts]
    answer_start_pred = sum(lens[:rank])
    answer_end_pred = sum(lens[:rank + 1])
    if answer_start >= answer_start_pred <= answer_end and answer_start_pred <= answer_end <= answer_end_pred:
        return True
    else:
        return False


class BaselineQA:
    def __init__(self):
        self.vectorizer = pickle.load(open(vector_file, mode='rb'))
        self.vocab = open(vocab_file, mode='r').read().splitlines()

    def evaluate(self, thresh=0.05):
        dataset = squad.Squad(train=True)
        prediction = []
        for index, [context, qas] in enumerate(dataset):
            if index % 100 == 0:
                print(index)
            contexts = []
            for sentence in sent_tokenizer.tokenize(context):
                sentence = tokenize(sentence)
                sentence = [word if in_vocab(self.vocab, word) else UNK for word in sentence]
                contexts.append(" ".join(sentence))

            context_vec = self.vectorizer.transform(contexts)
            for qa in qas:
                question, answer, answer_start, is_impossible = qa
                answer_end = answer_start + len(answer)

                question = [word if in_vocab(self.vocab, word) else UNK for word in tokenize(question)]
                question = " ".join(question)
                question_vec = self.vectorizer.transform([question])
                scores = [cosine_similarity(question_vec, vec).flatten() for vec in context_vec]
                scores = np.asarray(scores).flatten()

                ranks = np.argsort(scores)[::-1]

                if scores[ranks[0]] > thresh:
                    prediction.append(is_correct(contexts, ranks[0], answer_start, answer_end))
        accuracy = sum(prediction) / len(prediction)
        print(accuracy)


if __name__ == '__main__':
    # compute_vectors()

    print("Computing accuracy of the model") 
    baseline = BaselineQA()
    baseline.evaluate()
