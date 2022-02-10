import json
import os
import subprocess
import shutil
import time

import cv_svm_ocr
from io import BytesIO

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from functools import lru_cache

resources_path = 'images/chars2/'


def dump_index_itemid_relation():
    dump_data = {
        'idx2id': [],
        'id2idx': {}
    }
    collect_list = os.listdir(resources_path)
    collect_list.sort()
    index = 0
    for dirpath in collect_list:
        item_id = dirpath
        dump_data['idx2id'].append(item_id)
        dump_data['id2idx'][item_id] = index
        index += 1
    with open('index_itemid_relation.json', 'w') as f:
        json.dump(dump_data, f)
    return dump_data['idx2id'], dump_data['id2idx']


def show_img(img):
    cv2.imshow('test', img)
    cv2.waitKey()


def resize_char(img):
    h, w = img.shape[:2]
    scale = 16 / max(h, w)
    h = int(h * scale)
    w = int(w * scale)
    img2 = np.zeros((16, 16)).astype(np.uint8)
    img = cv2.resize(img, (w, h))

    img2[0:h, 0:w] = ~img
    # cv2.imshow('test', img2)
    # cv2.waitKey()
    return img2


def add_fake_char(img_map):
    # char '-'
    img_l = img_map.get('-')
    img_b = np.zeros((16, 16)).astype(np.uint8)
    img_w = 255 * np.ones((8, 16)).astype(np.uint8)
    img_b[0:8, 0:16] = img_w
    # cv2.imshow('test', img_b)
    # cv2.waitKey()
    img_l.append(img_b)


def load_images():
    img_map = {}
    collect_list = os.listdir(resources_path)
    collect_list.sort()
    weights = []
    for cdir in collect_list:
        dirpath = resources_path + cdir
        sub_dir_files = os.listdir(dirpath)
        weights.append(len(sub_dir_files))
        for filename in sub_dir_files:
            filepath = os.path.join(dirpath, filename)
            with open(filepath, 'rb') as f:
                nparr = np.frombuffer(f.read(), np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                image = resize_char(image)
                # image = np.expand_dims(image, 0)
                l = img_map.get(cdir, [])
                l.append(image)
                img_map[cdir] = l
    add_fake_char(img_map)
    return img_map


idx2id, id2idx = dump_index_itemid_relation()
img_map = load_images()
NUM_CLASS = len(idx2id)
print('NUM_CLASS', NUM_CLASS)


def add_noise(img, max_random_h):
    img = img.copy()
    h, w = img.shape
    count = np.random.randint(5, 10)
    for _ in range(count):
        x = np.random.randint(0, w)
        y = np.random.randint(0, max_random_h)
        img[y][x] = 255 * np.random.randint(0, 2)
    # print(img.shape)
    if np.random.randint(0, 2):
        scale = 40 / np.random.randint(30, 85)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        img = cv2.resize(img, (16, 16))
    h, w = img.shape
    t = 0
    l = np.random.randint(0, 2)
    r = np.random.randint(0, 2)
    b = np.random.randint(0, 2)
    img = img[t:h-b, l:w-r]
    # print(img.shape)
    img = cv2.resize(img, (16, 16))
    return img


def get_data():
    images = []
    labels = []
    for c in img_map.keys():
        cnt = 30 if c in '-5TR' else 10
        max_random_h = 6 if c == '-' else 16
        idxs = np.random.choice(range(len(img_map[c])), cnt)
        for idx in idxs:
            img = img_map[c][idx]
            image_aug = add_noise(img, max_random_h)
            # show_img(image_aug)
            image_aug = np.expand_dims(image_aug, 0)
            images.append(image_aug)
            labels.append(id2idx[c])
    images_np = np.stack(images, 0)
    labels_np = np.array(labels)

    # rand = np.random.RandomState(321)
    # shuffle = rand.permutation(len(images_np))
    # images_np = images_np[shuffle]
    # labels_np = labels_np[shuffle]

    # print(images_np.shape)
    return images_np, labels_np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),   # 16 * 16 * 16
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32 * 8 * 8
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(4, 4)  # 32 * 2 * 2
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 2 * 2, NUM_CLASS)
        )

    def forward(self, x):
        x = x / 255.
        x = x - 0.5
        out = self.conv(x)
        out = out.reshape(-1, 32 * 2 * 2)
        out = self.fc(out)

        return out


loss_func = nn.CrossEntropyLoss()
# loss_func = FocalLoss(NUM_CLASS)
# BCEWithLogitsLoss = nn.BCEWithLogitsLoss(weights_t)


def compute_loss(x, label):
    loss = loss_func(x, label)
    prec = (x.argmax(1) == label).float().mean().item()
    return loss, prec


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('train on:', device)
    model = Net().to(device)
    loss_func.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    step = 0
    prec = 0
    target_step = 3000
    best = 1
    saved = False
    while step < target_step or not saved:
        images_aug_np, label_np = get_data()
        images_aug = torch.from_numpy(images_aug_np).float().to(device)
        label = torch.from_numpy(label_np).long().to(device)
        optim.zero_grad()
        score = model(images_aug)
        loss, prec = compute_loss(score, label)
        loss.backward()
        optim.step()
        if step < 10 or step % 10 == 0:
            print(step, loss.item(), prec)
        step += 1
        if step > target_step - 500 and loss.item() < best and prec == 1:
            saved = True
            best = loss.item()
            print(f'save best {best}')
            torch.save(model.state_dict(), './model.pth')
            torch.onnx.export(model, images_aug, 'chars.onnx')
            # shutil.copyfile('chars.onnx', f'tmp/chars-{int(time.time())}.onnx')


@lru_cache(maxsize=1)
def load_model():
    model = Net()
    device = torch.device('cpu')
    model.load_state_dict(torch.load('./model.pth', map_location=device))
    model.eval()
    return model


def predict(img):
    model = load_model()
    char_imgs = cv_svm_ocr.crop_char_img(img)
    if not char_imgs:
        return ''
    roi_list = [np.expand_dims(resize_char(x), 0) for x in char_imgs]
    roi_np = np.stack(roi_list, 0)
    roi_t = torch.from_numpy(roi_np).float()
    with torch.no_grad():
        score = model(roi_t)
        probs = nn.Softmax(1)(score)
        predicts = score.argmax(1)

    probs = probs.cpu().data.numpy()
    predicts = predicts.cpu().data.numpy()
    # print([idx2id[p] for p in predicts], [probs[i, predicts[i]] for i in range(len(roi_list))])
    return ''.join([idx2id[p] for p in predicts])


@lru_cache(maxsize=1)
def load_onnx_model():
    return cv2.dnn.readNetFromONNX('chars.onnx')


def predict_cv(img):
    net = load_onnx_model()
    char_imgs = cv_svm_ocr.crop_char_img(img)
    if not char_imgs:
        return ''
    roi_list = [np.expand_dims(resize_char(x), 2) for x in char_imgs]
    blob = cv2.dnn.blobFromImages(roi_list)
    net.setInput(blob)
    score = net.forward()
    # probs = softmax(score)
    predicts = score.argmax(1)
    return ''.join([idx2id[p] for p in predicts])


def test_img_map(c):
    model = load_model()
    roi_list = img_map[c]
    cv2.imshow('test', np.transpose(roi_list[0], ))
    cv2.waitKey()
    roi_np = np.stack(roi_list, 0)
    roi_t = torch.from_numpy(roi_np).float()
    with torch.no_grad():
        score = model(roi_t)
        probs = nn.Softmax(1)(score)
        predicts = score.argmax(1)

    probs = probs.cpu().data.numpy()
    predicts = predicts.cpu().data.numpy()
    # print([idx2id[p] for p in predicts], [probs[i, predicts[i]] for i in range(len(roi_list))])
    return ''.join([idx2id[p] for p in predicts])


def screenshot():
    content = subprocess.check_output('adb exec-out "screencap -p"', shell=True)
    if os.name == 'nt':
        content = content.replace(b'\r\n', b'\n')
    # with open('images/screen.png', 'wb') as f:
    #     f.write(content)
    # img_array = np.asarray(bytearray(content), dtype=np.uint8)
    return Image.open(BytesIO(content))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


if __name__ == '__main__':
    # print(img_map.keys())
    train()
    # test()
    # prepare_train_resource()
    # export_onnx()
    # test_cv_onnx()
    # optimize_onnx()
    # test_img_map('B')
    # print(idx2id)
