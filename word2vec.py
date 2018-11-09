from collections import defaultdict

import dill
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


class word2vec():
    def __init__(self):
        self.n = settings['n']
        self.eta = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']
        pass

    # GENERATE TRAINING DATA
    def generate_training_data(self, settings, corpus):

        # GENERATE WORD COUNTS
        word_counts = defaultdict(int)
        for row in corpus:
            for word in row:
                word_counts[word] += 1

        self.v_count = len(word_counts.keys())

        # GENERATE LOOKUP DICTIONARIES
        self.words_list = sorted(list(word_counts.keys()), reverse=False)
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

        training_data = []
        # CYCLE THROUGH EACH SENTENCE IN CORPUS
        for sentence in corpus:
            sent_len = len(sentence)

            # CYCLE THROUGH EACH WORD IN SENTENCE
            for i, word in enumerate(sentence):

                # w_target  = sentence[i]
                w_target = self.word2onehot(sentence[i])

                # CYCLE THROUGH CONTEXT WINDOW
                w_context = []
                for j in range(i - self.window, i + self.window + 1):
                    if j != i and j <= sent_len - 1 and j >= 0:
                        w_context.append(self.word2onehot(sentence[j]))
                training_data.append([w_target, w_context])
        return np.array(training_data)

    # SOFTMAX ACTIVATION FUNCTION
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    # CONVERT WORD TO ONE HOT ENCODING
    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]
        word_index = self.word_index[word]
        word_vec[word_index] = 1
        return word_vec

    # FORWARD PASS
    def forward_pass(self, x):
        h = np.dot(self.w1.T, x)
        u = np.dot(self.w2.T, h)
        y_c = self.softmax(u)
        return y_c, h, u

    # BACKPROPAGATION
    def backprop(self, e, h, x):
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))

        # UPDATE WEIGHTS
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)
        pass

    # TRAIN W2V model
    def train(self, training_data):
        # INITIALIZE WEIGHT MATRICES
        self.w1 = np.random.uniform(-0.8, 0.8, (self.v_count, self.n))  # context matrix
        self.w2 = np.random.uniform(-0.8, 0.8, (self.n, self.v_count))  # embedding matrix

        # CYCLE THROUGH EACH EPOCH
        for i in range(0, self.epochs):

            self.loss = 0

            # CYCLE THROUGH EACH TRAINING SAMPLE
            for w_t, w_c in training_data:
                # FORWARD PASS
                y_pred, h, u = self.forward_pass(w_t)

                # CALCULATE ERROR
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)

                # BACKPROPAGATION
                self.backprop(EI, h, w_t)

                # CALCULATE LOSS
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
                self.loss += -2 * np.log(len(w_c)) - np.sum([u[word.index(1)] for word in w_c]) + (
                        len(w_c) * np.log(np.sum(np.exp(u))))

            print('EPOCH:', i, 'LOSS:', self.loss)
        return self.w1

    # input a word, returns a vector (if available)
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    # input a vector, returns nearest word(s)
    def vec_sim(self, vec, top_n):

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(vec, v_w2)
            theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), reverse=True)
        # words_sorted = sorted(word_sim.items(), key=lambda word, sim: sim, reverse=True)
        for word, sim in words_sorted[:top_n]:
            print('vec_sim', word, sim)

        pass

    # input word, returns top [n] most similar words
    def word_sim(self, word, top_n):

        w1_index = self.word_index[word]
        v_w1 = self.w1[w1_index]

        # CYCLE THROUGH VOCAB
        word_sim = {}
        for i in range(self.v_count):
            v_w2 = self.w1[i]
            theta_num = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_num / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda word, sim: sim, reverse=True)

        for word, sim in words_sorted[:top_n]:
            print('word_sim', word, sim)

        pass


with open('motion_capture_20181011-1931.dill', 'rb') as f:
    x = dill.load(f)
vec = [l[4] for l in x]
# print(len(vec))

x = map(str, vec)
x = list(x)


def remove_stop_words(corpus):
    stop_words = [', ']
    results = []
    for text in corpus:
        tmp = text.split(' ')
        for stop_word in stop_words:
            if stop_word in tmp:
                tmp.remove(stop_word)
        results.append(" ".join(tmp))
    print(results)
    return results


def vec_word(self, w2v, vec):
    w2v = word2vec()
    w2v.vec_sim(self, vec, top_n=3)


# x = remove_stop_words(x)
#X_train, X_test = train_test_split(x, test_size=0.33, shuffle=False)

settings = {}
settings['n'] = 2  # dimension of word embeddings
settings['window_size'] = 2  # context window +/- center word
settings['min_count'] = 0  # minimum word count
settings['epochs'] = 10000  # number of training epochs
settings['neg_samp'] = 10  # number of negative words to use during training
settings['learning_rate'] = 0.01  # learning rate
np.random.seed(0)  # set the seed for reproducibility

corpus = [x]

# INITIALIZE W2V MODEL
w2v = word2vec()

# generate training data
training_data = w2v.generate_training_data(settings, corpus)

# train word2vec model
w1 = w2v.train(training_data)
W1 = tf.Variable(w1, name="W1")
tf.add_to_collection('vars', W1)

print(W1.shape)
# save model
sess = tf.Session()
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    vectors = sess.run(tf.convert_to_tensor(w1))
    print(vectors)
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in path: %s" % save_path)

w2v_df = pd.DataFrame(vectors, columns=['x1', 'x2'])
print(w2v_df)

def vec2word(vector,first_n):
    w2v.vec_sim(vector, first_n)

# --- END ----------------------------------------------------------------------+
