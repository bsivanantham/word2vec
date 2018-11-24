import dill
import numpy as np


with open('motion_capture_20181011-1931.dill', 'rb') as f:
    x = dill.load(f)
vec = [l[4] for l in x]


arra =[]
x =[]
z =[]
flag = False
# vec = vec.tolist()
#vec = [[1,1,1,2],[1,1,1,2],[2,2,4,4,3],[2,2,4,4,3],[2,2,4,4,3],[1,1,1,2],[1,1,1,2],[1,1,1,2]]
# for i in range(len(vec)):
#     for j in range(len(vec)-1):
#         if((vec[i] != vec[j]).all()):
#             flag = True
#             #arra.append(vec[i])
#         else:
#             flag = False
#             #x.append(vec[i])
#     if(flag):
#         arra.append(vec[i-1])
#         arra.append(vec[i])
#     else:
#         x.append(vec[i])
d = []
count =0
# for i in range (len(vec)):
#     for j in range(len(vec)-1):
#         arra = vec[i]
#         if((arra == vec[j]).all()):
#             count = count+1
#         else:
#             arra = vec[j]
#     if (count> 2):
#         z.append(vec[i])
#     else:
#         x.append(vec[i])
arr = vec[0]
z.append(vec[0])
for i in range (len(vec)):
    if((arr != vec[i]).any()):
        z.append(vec[i])
        arr = vec[i]

# vec = [(list(y)) for y in set(tuple(x) for x in vec)]
#
# vec = [list(x) for x in vec]
z = np.array(z)
z = np.round(z)
print(len(vec))
print(z)
with open("resultround.dill", "wb") as dill_file:
    dill.dump(z, dill_file)