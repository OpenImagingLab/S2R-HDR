import numpy as np
import random


def random_crop(inputs, size, margin=0):
    is_list = type(inputs) == list 
    if not is_list: inputs = [inputs]

    outputs = []
    h, w, _ = inputs[0].shape
    c_h, c_w = size
    if h != c_h or w != c_w:
        t = random.randint(0+margin, h - c_h-margin)
        l = random.randint(0+margin, w - c_w-margin)
        for img in inputs:
            outputs.append(img[t: t+c_h, l: l+c_w])
    else:
        outputs = inputs
    if not is_list: outputs = outputs[0]
    return outputs

def random_crop_v2(inputs1, inputs2, size, margin=0):
    is_list = type(inputs1) == list 
    if not is_list: 
        inputs1 = [inputs1]
        inputs2 = [inputs2]
    outputs1 = []
    outputs2 = []
    h, w, _ = inputs1[0].shape
    c_h, c_w = size
    if h != c_h or w != c_w:
        t = random.randint(0+margin, h - c_h-margin)
        l = random.randint(0+margin, w - c_w-margin)
        for img1, img2 in zip(inputs1, inputs2):
            outputs1.append(img1[t: t+c_h, l: l+c_w])
            outputs2.append(img2[t: t+c_h, l: l+c_w])
    else:
        outputs1 = inputs1
        outputs2 = inputs2
    if not is_list: 
        outputs1 = outputs1[0]
        outputs2 = outputs2[0]
    return outputs1, outputs2


def random_crop_v4(inputs1, inputs2, inputs3, inputs4, size, margin=0):
    is_list = type(inputs1) == list 
    if not is_list: 
        inputs1 = [inputs1]
        inputs2 = [inputs2]
        inputs3 = [inputs3]
        inputs4 = [inputs4]
    outputs1 = []
    outputs2 = []
    outputs3 = []
    outputs4 = []
    h, w, _ = inputs1[0].shape
    c_h, c_w = size
    if h != c_h or w != c_w:
        t = random.randint(0+margin, h - c_h-margin)
        l = random.randint(0+margin, w - c_w-margin)
        for img1, img2, img3, img4 in zip(inputs1, inputs2, inputs3, inputs4):
            outputs1.append(img1[t: t+c_h, l: l+c_w])
            outputs2.append(img2[t: t+c_h, l: l+c_w])
            outputs3.append(img3[t: t+c_h, l: l+c_w])
            outputs4.append(img4[t: t+c_h, l: l+c_w])
    else:
        outputs1 = inputs1
        outputs2 = inputs2
        outputs3 = inputs3
        outputs4 = inputs4
    if not is_list: 
        outputs1 = outputs1[0]
        outputs2 = outputs2[0]
        outputs3 = outputs3[0]
        outputs4 = outputs4[0]
    return outputs1, outputs2, outputs3, outputs4


def center_crop(inputs, size, margin=0):
    is_list = type(inputs) == list 
    if not is_list: inputs = [inputs]

    outputs = []
    h, w, _ = inputs[0].shape
    c_h, c_w = size
    if h != c_h or w != c_w:
        t = (h - c_h - margin) // 2
        l = (w - c_w - margin) // 2
        for img in inputs:
            outputs.append(img[t: t+c_h, l: l+c_w])
    else:
        outputs = inputs
    if not is_list: outputs = outputs[0]
    return outputs


def random_flip_lrud(inputs):
    is_list = type(inputs) == list 
    if not is_list: inputs = [inputs]

    outputs = []
    horizon_flip = True if np.random.random() > 0.5 else False
    vertical_flip = True if np.random.random() > 0.5 else False # vertical flip
    for img in inputs:
        if horizon_flip:
            img = np.fliplr(img)
        if vertical_flip:
            img = np.flipud(img)
        outputs.append(img.copy())
    if not is_list: outputs = outputs[0]
    return outputs

def random_flip_lrud_v2(inputs1, inputs2):
    is_list = type(inputs1) == list 
    if not is_list: 
        inputs1 = [inputs1]
        inputs2 = [inputs2]
    outputs1 = []
    outputs2 = []
    horizon_flip = True if np.random.random() > 0.5 else False
    vertical_flip = True if np.random.random() > 0.5 else False # vertical flip
    for img1, img2 in zip(inputs1, inputs2):
        if horizon_flip:
            img1 = np.fliplr(img1)
            img2 = np.fliplr(img2)
        if vertical_flip:
            img1 = np.flipud(img1)
            img2 = np.flipud(img2)
        outputs1.append(img1.copy())
        outputs2.append(img2.copy())
    if not is_list: 
        outputs1 = outputs1[0]
        outputs2 = outputs2[0]
    return outputs1, outputs2

def random_flip_lrud_v4(inputs1, inputs2, inputs3, inputs4):
    is_list = type(inputs1) == list 
    if not is_list: 
        inputs1 = [inputs1]
        inputs2 = [inputs2]
        inputs3 = [inputs3]
        inputs4 = [inputs4]
    outputs1 = []
    outputs2 = []
    outputs3 = []
    outputs4 = []
    horizon_flip = True if np.random.random() > 0.5 else False
    vertical_flip = True if np.random.random() > 0.5 else False # vertical flip
    for img1, img2, img3, img4 in zip(inputs1, inputs2, inputs3, inputs4):
        if horizon_flip:
            img1 = np.fliplr(img1)
            img2 = np.fliplr(img2)
            img3 = np.fliplr(img3)
            img4 = np.fliplr(img4)
        if vertical_flip:
            img1 = np.flipud(img1)
            img2 = np.flipud(img2)
            img3 = np.flipud(img3)
            img4 = np.flipud(img4)
        outputs1.append(img1.copy())
        outputs2.append(img2.copy())
        outputs3.append(img3.copy())
        outputs4.append(img4.copy())
    if not is_list: 
        outputs1 = outputs1[0]
        outputs2 = outputs2[0]
        outputs3 = outputs3[0]
        outputs4 = outputs4[0]
    return outputs1, outputs2, outputs3, outputs4
