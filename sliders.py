from guizero import App, Slider, TextBox
import _pickle
import numpy as np

slider1value = 0
slider2value = 0
def slider_changed1(slider_value):
    textbox1.value = slider_value
    slider1value =slider_value

def slider_changed2(slider_value):
    textbox2.value = slider_value
    slider2value = slider_value

# input a vector, returns nearest word(s)
def vec_sim(vec, top_n):

    # CYCLE THROUGH VOCAB
    word_sim = {}
    for i in range(self.v_count):
        v_w2 = self.w1[i]
        theta_num = np.dot(vec, v_w2)
        theta_den = np.linalg.norm(vec) * np.linalg.norm(v_w2)
        theta = theta_num / theta_den

        word = self.index_word[i]
        word_sim[word] = theta

    words_sorted = sorted(word_sim.items(), key=lambda word, sim: sim, reverse=True)
    for word, sim in words_sorted[:top_n]:
        print('vec_sim', word, sim)

    pass

app = App(title="Reinforcement Learning")
slider1 = Slider(app, command=slider_changed1)
textbox1 = TextBox(app)

slider2 = Slider(app, command=slider_changed2)
textbox2 = TextBox(app)

app.display()

filename = 'savedmode.sav'
word2vec =_pickle.load(open(filename,'rb'))
word_sim = {}
vec = list()
vec.append(slider1value)
vec.append(slider2value)
word2vec.vec_word(self,vec,5)
