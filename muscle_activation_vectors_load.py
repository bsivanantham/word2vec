import dill

with open('motion_capture_20181011-1931.dill', 'rb') as f:
    x = dill.load(f)
vec = [l[4] for l in x]

#print(vec[1])
#print(vec[2])
print(len(vec))

arra =[]
#
# for i in range(len(vec)):
#     for j in range(len(vec)-1):
#         if((vec[i] != vec[j]).all()):
#             print("working")
#         else:
#             arra.append(vec[i])
#             print(vec[i])

import numpy as np
#
# lout2=[np.unique(vec)
# print(len(lout2))

# vec = vec.tolist()
vec = [x.tolist() for x in vec]
vec = [(list(y)) for y in set(tuple(x) for x in vec)]
# vec = [list(x) for x in vec]
# vec = np.array(vec)
print(len(vec))

with open("resultstatsDF.dill", "wb") as dill_file:
    dill.dump(vec, dill_file)