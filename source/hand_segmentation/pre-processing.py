# Paths
from paths import *

# Plots
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random

import cv2
import argparse
from scipy import ndimage


parser = argparse.ArgumentParser()
parser.add_argument("--set1", type=str)
parser.add_argument("--set2", type=str)


args = parser.parse_args()
fset1 = args.set1
fset2 = args.set2

set1 = np.load(fset1)
set2 = np.load(fset2)

nsize = set1.shape[1]

print("Shifting and rotating")
for i in range(0, int(set1.shape[0])):
    print(i)
    # random rotation
    deg = random.randint(0, 359)
    set1[i, :, :, :] = ndimage.rotate(set1[i, :, :, :], deg, reshape=False)
    set2[i, :, :] = ndimage.rotate(set2[i, :, :], deg, reshape=False)

    # random shift
    s = np.float32([[1, 0, random.randint(0, 20) - 10], [0, 1, random.randint(0, 20) - 10]])
    set1[i, :, :, :] = cv2.warpAffine(set1[i, :, :, :], s, (nsize, nsize))
    set2[i, :, :] = cv2.warpAffine(set2[i, :, :], s, (nsize, nsize))

    # random flip lr
    if bool(random.getrandbits(1)):
        set1[i, :, :, :] = np.fliplr(set1[i, :, :, :])
        set2[i, :, :] = np.fliplr(set2[i, :, :])

# Correct errors introduced by rotation
for k in range(set1.shape[0]):
    print("Correcting")
    print(k)
    for i in range(256):
        for j in range(256):
            if set2[k,i,j] >= 8:
                set2[k,i,j] = set2[k,i-1,j]

print("Now saving")

np.save(fset1+"processed", set1)
np.save(fset2+"processed", set2)
