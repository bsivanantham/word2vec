import dill
import numpy as np
import tensorflow as tf
from collections import defaultdict


from sklearn.model_selection import train_test_split

with open('motion_capture_20181011-1931.dill', 'rb') as f:
    x = dill.load(f)
vec = [l[4] for l in x]
# print(len(vec))

x = map(str, vec)
x = list(x)

X_train, X_test = train_test_split(x, test_size=0.33, shuffle=False)

corpus = [X_test]

#restore model for testing
sess = tf.Session()
new_saver = tf.train.import_meta_graph('model.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.get_collection('vars')
for v in all_vars:
    w1 = sess.run(v)
    print(w1)

#generate data for testing
word_counts = defaultdict(int)
for row in corpus:
    for word in row:
        word_counts[word] += 1

v_count = len(word_counts.keys())

# GENERATE LOOKUP DICTIONARIES
words_list = sorted(list(word_counts.keys()), reverse=False)
word_index = dict((word, i) for i, word in enumerate(words_list))
index_word = dict((i, word) for i, word in enumerate(words_list))


def vec_sim(vec, top_n):
    # CYCLE THROUGH VOCAB
    word_sim = {}
    output = []
    for i in range(v_count):
        v_w2 = w1[i]
        theta_num = np.dot(vec, v_w2)
        theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
        theta = theta_num / theta_den

        word = index_word[i]
        word_sim[word] = theta

    words_sorted = sorted(word_sim.items(), reverse=True)
    # words_sorted = sorted(word_sim.items(), key=lambda word, sim: sim, reverse=True)
    for word, sim in words_sorted[:top_n]:
        print('vec_sim', word, sim)
        output.append(word)
        output.append(sim)

    return output

corpus = [(1,1)]
output = vec_sim(corpus,1)
print(output)