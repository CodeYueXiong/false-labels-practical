#----------------------------------------------------#
#   Get the ground-truth of the test set
#----------------------------------------------------#

import sys
import os
import glob
import numpy as np
# import xml.etree.ElementTree as ET
global test_raw_anno
test_raw_anno = np.load("Test.npy", allow_pickle=True).item()

#---------------------------------------------------#
#   get label classes
#---------------------------------------------------#


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

image_ids = open('./ImageSets/Main/test.txt').read().strip().split()

if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/ground-truth"):
    os.makedirs("./input/ground-truth")

for image_id in image_ids:
    with open("./input/ground-truth/"+image_id+".txt", "w") as new_f:
        obj_name = test_raw_anno[image_id][1]['Class']

        xmax, ymax = np.max(test_raw_anno[image_id][3]['Annotations'],axis=0)[0], np.max(test_raw_anno[image_id][3]['Annotations'],axis=0)[1]
        xmin, ymin = np.min(test_raw_anno[image_id][3]['Annotations'],axis=0)[0], np.min(test_raw_anno[image_id][3]['Annotations'],axis=0)[1]
        left, top, right, bottom = xmin, ymin, xmax, ymax

        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

print("Conversion completed!")