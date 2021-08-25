import numpy as np
import torch
from PIL import Image


def point_form(boxes):
    # ------------------------------#
    # left uppermost and right bottom
    # ------------------------------#
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                      boxes[:, :2] + boxes[:, 2:] / 2), 1)


def center_size(boxes):
    # ------------------------------#
    #   center and w, h
    # ------------------------------#
    return torch.cat((boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2], 1)


def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    # -----------------------------------------------#
    #   finding the left uppermost of intersection
    # -----------------------------------------------#
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    # -----------------------------------------------#
    #   finding the right bottom of intersection
    # -----------------------------------------------#
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp((max_xy - min_xy), min=0)
    # ----------------------------------------------------------------------#
    #   Calculate the overlap area of the prior frame and all real frames
    # ----------------------------------------------------------------------#
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    # -----------------------------------------------#
    #   The shape of the returned 'inter' is [A,B]
    #   which represents the intersection rectangle
    #   for each real box and a priori box
    # -----------------------------------------------#
    inter = intersect(box_a, box_b)
    # -------------------------------------#
    #   Calculate the respective areas of
    #   the prior box and the ground truth box
    # -------------------------------------#
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

    union = area_a + area_b - inter
    # -------------------------------------#
    #   The intersection ratio of each
    #   ground truth box and a priori box is [A,B]
    # -------------------------------------#
    return inter / union


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    # ----------------------------------------------#
    #   Calculate the degree of intersection of all
    #   prior boxes and ground truth boxes
    # ----------------------------------------------#
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # ----------------------------------------------#
    #   The best intersection ratio of all true boxes and a priori boxes
    #   best_prior_overlap [truth_box,1]
    #   best_prior_idx [truth_box,0]
    # ----------------------------------------------#
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    # ----------------------------------------------#
    #   The best intersection ratio of all priori boxes and ground truth boxes
    #   best_truth_overlap [1,prior]
    #   best_truth_idx [1,prior]
    # ----------------------------------------------#
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)

    # ----------------------------------------------#
    #   Used to ensure that each real box has at
    #   least one corresponding a priori box
    # ----------------------------------------------#
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)

    # ----------------------------------------------#
    #   Get the ground truth box corresponding to
    #   each a priori box [num_priors,4]
    # ----------------------------------------------#
    matches = truths[best_truth_idx]
    # Shape: [num_priors]
    conf = labels[best_truth_idx] + 1

    # ----------------------------------------------#
    #   If the intersection ratio is less than
    #   threshold, it is considered background
    # ----------------------------------------------#
    conf[best_truth_overlap < threshold] = 0

    # ----------------------------------------------#
    #   Encode using ground truth box and priori box
    #   The encoded result is the predicted result
    #   that the network should have
    # ----------------------------------------------#
    loc = encode(matches, priors, variances)

    # [num_priors,4]
    loc_t[idx] = loc
    # [num_priors]
    conf_t[idx] = conf


def encode(matched, priors, variances):
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])

    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)


# -------------------------------------------------------------------#
#   Adapted from https://github.com/Hakuyume/chainer-ssd
#   Use the prediction results to adjust the prior frame, the first two parameters are used to adjust the xy axis coordinates of the center
#   The last two parameters are used to adjust the width and height of the prior frame
# -------------------------------------------------------------------#
def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


# -------------------------------------------------------------------#
#   Original author: Francisco Massa:
#   https://github.com/fmassa/object-detection.torch
#   Ported to PyTorch by Max deGroot (02/01/2017)
#  This part is used to perform non-maximum suppression, that is,
#  to screen out the box with the largest score in a certain area.
# -------------------------------------------------------------------#
def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union
        idx = idx[IoU.le(overlap)]
    return keep, count


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def ssd_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = np.concatenate(((top + bottom) / 2, (left + right) / 2), axis=-1)
    box_hw = np.concatenate((bottom - top, right - left), axis=-1)

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes