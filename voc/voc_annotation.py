# ---------------------------------------------#
# please make sure to modify the classes
# as the train.txt might be wrong
# ---------------------------------------------#
import math
import random
import numpy as np

sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
# -----------------------------------------------------#
#   the classes must correspond to the classes.txt
# -----------------------------------------------------#
classes = ["Blob", "Blur", "Distortion", "Channel_Change"]

train_cls = []
val_cls = []
test_cls = []

global train_anno, val_anno, test_anno
train_anno = np.load("Train.npy", allow_pickle=True).item()
val_anno = np.load("Val.npy", allow_pickle=True).item()
test_anno = np.load("Test.npy", allow_pickle=True).item()

# only used for train dataset
def make_false_labels(false_labels_ratio):
    train_label_names = []
    train_img_ids = []
    train_label_dict = {}
    train_anno = np.load("Train.npy", allow_pickle=True).item()
    classes = ["Blob", "Blur", "Distortion", "Channel_Change"]

    for idx in range(len(train_anno)):
        img_id = train_anno[list(train_anno)[idx]][0]['Img_id']  # get the image ids
        img_class = train_anno[list(train_anno)[idx]][1]['Class']  # get the image classes
        train_img_ids.append(img_id)
        train_label_names.append(img_class)
        train_label_dict[img_id] = img_class

    false_labels_count = math.floor(len(train_label_names) * false_labels_ratio)
    print("Number of false labels:", false_labels_count)
    wrong_label_ids = random.sample(range(0, len(train_label_names)), false_labels_count)

    # Manipulate label if index in wrong_label_ids
    for i in range(0, len(train_label_names)):
        train_classes = classes.copy()
        if i in wrong_label_ids:
            # Get index of true label
            true_label_index = train_anno[list(train_anno)[i]][1]['Class']
            image_id = train_anno[list(train_anno)[i]][0]['Img_id']
            # Get indices of all false labels (=all except trube label index)
            train_classes.remove(true_label_index)
            false_label_indices = train_classes

            # new label prepared
            new_label_index = random.sample(set(false_label_indices), 1)  # wrong with set
            # Set randomly chosen label value to 1
            d = {image_id: new_label_index[0]}  # remember to index with 0
            train_label_dict.update(d)

    return train_label_dict

train_falselabel_alldict = make_false_labels(0.1)

global train_falselabels_alldict
train_falselabels_alldict = train_falselabel_alldict

# only used for train dataset
def make_false_labels(false_labels_ratio):
    train_label_names = []
    train_img_ids = []
    train_label_dict = {}
    train_anno = np.load("Train.npy", allow_pickle=True).item()
    classes = ["Blob", "Blur", "Distortion", "Channel_Change"]

    for idx in range(len(train_anno)):
        img_id = train_anno[list(train_anno)[idx]][0]['Img_id']  # get the image ids
        img_class = train_anno[list(train_anno)[idx]][1]['Class']  # get the image classes
        train_img_ids.append(img_id)
        train_label_names.append(img_class)
        train_label_dict[img_id] = img_class

    false_labels_count = math.floor(len(train_label_names) * false_labels_ratio)
    print("Number of false labels:", false_labels_count)
    wrong_label_ids = random.sample(range(0, len(train_label_names)), false_labels_count)

    # Manipulate label if index in wrong_label_ids
    for i in range(0, len(train_label_names)):
        train_classes = classes.copy()
        if i in wrong_label_ids:
            # Get index of true label
            true_label_index = train_anno[list(train_anno)[i]][1]['Class']
            image_id = train_anno[list(train_anno)[i]][0]['Img_id']
            # Get indices of all false labels (=all except trube label index)
            train_classes.remove(true_label_index)
            false_label_indices = train_classes

            # new label prepared
            new_label_index = random.sample(set(false_label_indices), 1)  # wrong with set
            # Set randomly chosen label value to 1
            d = {image_id: new_label_index[0]}  # remember to index with 0
            train_label_dict.update(d)

    return train_label_dict


def convert_annotation(image_id, list_file, split):
    if split == "train":
        # train_anno = np.load("Train.npy", allow_pickle=True).item()
        # for idx in range(len(train_anno)):
        train_labels = train_falselabels_alldict[image_id]
        xmax, ymax = np.max(train_anno[image_id][3]['Annotations'], axis=0)[0], \
                     np.max(train_anno[image_id][3]['Annotations'], axis=0)[1]
        xmin, ymin = np.min(train_anno[image_id][3]['Annotations'], axis=0)[0], \
                     np.min(train_anno[image_id][3]['Annotations'], axis=0)[1]
        # train_cls.append(train_labels)

        cls_id = ""
        cls_ids = []
        if train_labels == "Blob":
            cls_id = "0"
        elif train_labels == "Blur":
            cls_id = "1"
        elif train_labels == "Distortion":
            cls_id = "2"
        elif train_labels == "Channel_Change":
            cls_id = "3"
        list_file.write(" " + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + "," + cls_id)

    elif split == "val":
        # val_anno = np.load("Val.npy", allow_pickle=True).item()
        val_labels = val_anno[image_id][1]['Class']
        xmax, ymax = np.max(val_anno[image_id][3]['Annotations'], axis=0)[0], \
                     np.max(val_anno[image_id][3]['Annotations'], axis=0)[1]
        xmin, ymin = np.min(val_anno[image_id][3]['Annotations'], axis=0)[0], \
                     np.min(val_anno[image_id][3]['Annotations'], axis=0)[1]

        cls_id = ""
        cls_ids = []
        if val_labels == "Blob":
            cls_id = "0"
        elif val_labels == "Blur":
            cls_id = "1"
        elif val_labels == "Distortion":
            cls_id = "2"
        elif val_labels == "Channel_Change":
            cls_id = "3"
        list_file.write(" " + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + "," + cls_id)

    elif split == "test":
        # test_anno = np.load("Test.npy", allow_pickle=True).item()
        # for idx in range(len(test_anno)):
        test_labels = test_anno[image_id][1]['Class']
        xmax, ymax = np.max(test_anno[image_id][3]['Annotations'], axis=0)[0], \
                     np.max(test_anno[image_id][3]['Annotations'], axis=0)[1]
        xmin, ymin = np.min(test_anno[image_id][3]['Annotations'], axis=0)[0], \
                     np.min(test_anno[image_id][3]['Annotations'], axis=0)[1]

        cls_id = ""
        if test_labels == "Blob":
            cls_id = "0"
        elif test_labels == "Blur":
            cls_id = "1"
        elif test_labels == "Distortion":
            cls_id = "2"
        elif test_labels == "Channel_Change":
            cls_id = "3"
        list_file.write(" " + str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + "," + cls_id)

train_img_ids = open('./ImageSets/Main/train.txt', encoding='utf-8').read().strip().split()
train_list_files = open('2007_train.txt', 'w', encoding='utf-8')
for image_id in train_img_ids:
    train_list_files.write('/content/drive/MyDrive/falselabels/data/Train/%s.png' % (image_id))
    convert_annotation(image_id, train_list_files, split="train")
    train_list_files.write('\n')
train_list_files.close()

val_img_ids = open('./ImageSets/Main/val.txt', encoding='utf-8').read().strip().split()
val_list_files = open('2007_val.txt', 'w', encoding='utf-8')
for image_id in val_img_ids:
        val_list_files.write('/content/drive/MyDrive/falselabels/data/Val/%s.png'%(image_id))
        convert_annotation(image_id, val_list_files, split="val")
        val_list_files.write('\n')
val_list_files.close()

test_img_ids = open('./ImageSets/Main/test.txt', encoding='utf-8').read().strip().split()
test_list_files = open('2007_test.txt', 'w', encoding='utf-8')
for image_id in test_img_ids:
        test_list_files.write('/content/drive/MyDrive/falselabels/data/Test/%s.png'%(image_id))
        convert_annotation(image_id, test_list_files, split="test")
        test_list_files.write('\n')
test_list_files.close()