from collections import Counter

from nltk import WordPunctTokenizer, WordNetLemmatizer, PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import squad

tokenizer = WordPunctTokenizer()
lemmatizer = WordNetLemmatizer()
sent_tokenizer = PunktSentenceTokenizer()


class BaselineQA:
    def __init__(self):
        dataset = squad.Squad(train=True)
        vocab_size = 10000

        text = [context for context, qas in dataset]
        text = " ".join(text)
        sentences = sent_tokenizer.tokenize(text)

        words = tokenizer.tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words]
        words = Counter(words).most_common(vocab_size)
        vocab = [word for (word, count) in words]
        vocab.append('unk')
        print("Vocab size : ", len(vocab))

        # self.word2index = dict(zip(vocab, list(range(len(vocab)))))
        # self.index2word = dict(zip(list(range(len(vocab))), vocab))

        self.vectorizer = TfidfVectorizer().fit(sentences)

    def evaluate(self, thresh=0.5):
        dataset = squad.Squad(train=True)
        for context, qas in dataset:
            sentences = sent_tokenizer.tokenize(context)
            context_vec = self.vectorizer.transform(sentences)
            for qa in qas:
                question, answer, answer_start, is_impossible = qa
                question_vec = self.vectorizer.transform([question])
                scores = [cosine_similarity(question_vec, vec) for vec in context_vec]

                print(question, '\n', sentences)
                print("Scores : ", scores)
                ranks = np.argsort(scores)
                print("Ranks : ", ranks)
                if scores[ranks[0]] > thresh:
                    # We have an answer for this question in this context paragraph
                    print(question, sentences[ranks[0]])
            break


if __name__ == '__main__':
    baseline = BaselineQA()
    baseline.evaluate()
