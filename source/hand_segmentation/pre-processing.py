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
parser.add_argument("--set1")
parser.add_argument("--set2")


args = parser.parse_args()
fset1 = args.set1
fset2 = args.set2

set1 = np.load(fset1)
set2 = np.load(fset2)

nsize = set.shape[0]

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

    # random flip ud
    if bool(random.getrandbits(1)):
        set1[i, :, :, :] = np.flipud(set1[i, :, :, :])
        set2[i, :, :] = np.flipud(set2[i, :, :])