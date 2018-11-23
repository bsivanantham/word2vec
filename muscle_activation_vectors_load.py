import dill
import numpy as np


with open('motion_capture_20181011-1931.dill', 'rb') as f:
    x = dill.load(f)
vec = [l[4] for l in x]

print(len(vec))

# vec = vec.tolist()
vec = [x.tolist() for x in vec]
vec = [(list(y)) for y in set(tuple(x) for x in vec)]

vec = [list(x) for x in vec]
vec = np.array(vec)
vec = np.round(vec)
print(len(vec))

with open("resultround.dill", "wb") as dill_file:
    dill.dump(vec, dill_file)