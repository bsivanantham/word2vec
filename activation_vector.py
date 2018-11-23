import dill
import numpy as np

with open('resultstatsDF.dill', 'rb') as f:
    xy = dill.load(f)


#print(vec[1])
#myList = [round(x) for x in xy]
#print([round(x) for x in xy])

vec = [[np.round(i) for i in nested] for nested in xy]

# with open("intconverted.dill", "wb") as dill_file:
#     dill.dump(vec, dill_file)
print(vec)