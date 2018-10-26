import dill
with open('motion_capture_20181011-1931.dill', 'rb') as f:
    x = dill.load(f)
vec = [l[4] for l in x]