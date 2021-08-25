# ----------------------------------------------------------------------#
#   The train/test splitting is executed in train.py
#   It is normal for val and test.txt files to be empty as not
#   needed in training.
# ----------------------------------------------------------------------#

import os
import random
import numpy as np

random.seed(0)

trainNumpyPath = r"Train.npy"
valNumpyPath = r"Val.npy"
testNumpyPath = r"Test.npy"
saveBasePath = r"./ImageSets/Main/"

train_raw = np.load(trainNumpyPath, allow_pickle=True).item()
val_raw = np.load(valNumpyPath, allow_pickle=True).item()
test_raw = np.load(testNumpyPath, allow_pickle=True).item()

train_name_list = []
val_name_list = []
test_name_list = []

ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')


def load_name(raw_file, split):
    if split == "train":
        for idx in range(len(raw_file)):
            img_id = raw_file[list(raw_file)[idx]][0]['Img_id'] + '\n'
            ftrain.write(img_id)
    elif split == "val":
        for idx in range(len(raw_file)):
            img_id = raw_file[list(raw_file)[idx]][0]['Img_id'] + '\n'
            fval.write(img_id)
    elif split == "test":
        for idx in range(len(raw_file)):
            img_id = raw_file[list(raw_file)[idx]][0]['Img_id'] + '\n'
            ftest.write(img_id)


load_name(train_raw, split="train")
load_name(val_raw, split="val")
load_name(test_raw, split="test")

# ftrainval.close()
ftrain.close()
fval.close()
ftest.close()