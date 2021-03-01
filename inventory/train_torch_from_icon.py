import time
from io import BytesIO

import numpy as np
import cv2
import os
import subprocess
from PIL import Image


import inventory
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import json


icon_files = list(filter(lambda x: x.split('.')[0].isdigit(), os.listdir('images/icon/')))
img_size = 32
collect_path = 'images/collect/'


def dump_index_itemid_relation():
    dump_data = {
        'idx2id': [],
        'id2idx': {}
    }
    index = 0
    for file in icon_files:
        item_id = file.split('.')[0]
        dump_data['idx2id'].append(item_id)
        dump_data['id2idx'][item_id] = index
        index += 1
    dump_data['idx2id'].append('other')
    dump_data['id2idx']['other'] = index
    with open('index_itemid_relation.json', 'w') as f:
        json.dump(dump_data, f)
    return dump_data['idx2id'], dump_data['id2idx']


def load_images():
    img_map = {}

    for file in icon_files:
        image = cv2.imread(f'images/icon/{file}', cv2.IMREAD_UNCHANGED)

        # image = cv2.resize(image, (128, 128))
        img_map[file] = image

    return img_map


idx2id, id2idx = dump_index_itemid_relation()
NUM_CLASS = len(idx2id)
print('NUM_CLASS', NUM_CLASS)
img_map = load_images()


def crop_item_middle_img(cv_item_img, ox, oy, radius):
    ratio = radius / 60
    y1 = int(oy - (40 * ratio))
    y2 = int(oy + (24 * ratio))
    x1 = int(ox - (32 * ratio))
    x2 = int(ox + (32 * ratio))
    return cv2.resize(cv_item_img[y1:y2, x1:x2], (64, 64))


def get_data():
    images = []
    labels = []
    for file in icon_files:
        item_id = file[:-4]

        image = img_map[file]

        item_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        circles = inventory.get_circles(item_gray, 75, 100)
        c = circles[0]
        ox = c[0] + np.random.randint(-2, 2)
        oy = c[1] + np.random.randint(-2, 2)
        img = crop_item_middle_img(image, ox, oy, c[2])
        # inventory.show_img(img)

        image_aug = img

        # inventory.show_img(image_aug)

        image_aug = image_aug[..., :-1]

        # image_aug = image_aug / 255 * 2 - 1

        images.append(image_aug)
        labels.append(id2idx[item_id])
    images_np = np.transpose(np.stack(images, 0), [0, 3, 1, 2])
    labels_np = np.array(labels)
    # print(images_np.shape)
    # imgs, lbls = get_noise_data()
    # images_np = np.concatenate((images_np, imgs))
    # labels_np = np.concatenate((labels_np, lbls))
    # print(images_np.shape)
    # exit()
    return images_np, labels_np


def get_noise_data():
    dummy_size = 200
    images_np = np.random.rand(dummy_size, 3, 64, 64)
    labels_np = np.asarray([NUM_CLASS - 1]).repeat(dummy_size)
    return images_np, labels_np


class Cnn(nn.Module):
    def __init__(self):  # 3 * 64 * 64
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 3, stride=1, padding=1),  # 6 * 64 * 64
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 6 * 32 * 32
            nn.Conv2d(6, 16, 5, stride=1, padding=0),  # 16 * 28 *28
            nn.ReLU(True),
            nn.MaxPool2d(4, 4))  # # 16 * 7 * 7

        self.fc = nn.Sequential(
            nn.Linear(784, 400),  # 784 = 16 * 7 * 7
            nn.ReLU(True),
            nn.Linear(400, 120),
            nn.ReLU(True),
            nn.Linear(120, NUM_CLASS))

    def forward(self, x):
        out = self.conv(x)
        out = out.reshape(-1, 16 * 7 * 7)  # 784 = 16 * 7 * 7
        out = self.fc(out)
        return out


def compute_loss(x, label):
    loss = nn.CrossEntropyLoss()(x, label)
    prec = (x.argmax(1) == label).float().mean()
    return loss, prec


def train():
    model = Cnn().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(500):
        images_aug_np, label_np = get_data()
        images_aug = torch.from_numpy(images_aug_np).float().cuda()
        label = torch.from_numpy(label_np).long().cuda()
        optim.zero_grad()
        score = model(images_aug)
        loss, prec = compute_loss(score, label)
        loss.backward()
        optim.step()
        if step < 10 or step % 10 == 0:
            print(step, loss.item(), prec.item())
    torch.save(model.state_dict(), './model2.bin')


def load_model():
    model = Cnn()
    device = torch.device('cpu')
    model.load_state_dict(torch.load('./model2.bin', map_location=device))
    model.eval()
    return model


def predict(model, roi_list):
    roi_np = np.stack(roi_list, 0)
    roi_t = torch.from_numpy(roi_np).float()
    with torch.no_grad():
        score = model(roi_t)
        probs = nn.Softmax(1)(score)
        predicts = score.argmax(1)

    probs = probs.cpu().data.numpy()
    predicts = predicts.cpu().data.numpy()
    return [idx2id[p] for p in predicts], [probs[i, predicts[i]] for i in range(len(roi_list))]


def test():

    model = load_model()
    screen = Image.open('images/screen2.png')
    items = inventory.get_all_item_img_in_screen(screen)
    roi_list = []
    for x in items:
        roi = x['rectangle2'].copy()
        # inventory.show_img(roi)

        roi = np.transpose(roi, (2, 0, 1))
        roi_list.append(roi)
    res = predict(model, roi_list)
    print(res)
    for i in range(len(res[0])):
        item_id = res[0][i]
        if item_id == 'other':
            print(res[1][i], 'other')
        else:
            print(res[1][i], inventory.item_map[int(item_id)])
        inventory.show_img(items[i]['rectangle2'])
    roi_np = np.stack(roi_list, 0)
    roi_t = torch.from_numpy(roi_np).float()

    torch.onnx.export(model, roi_t, 'ark_material.onnx')


def screenshot():
    content = subprocess.check_output('adb exec-out "screencap -p"', shell=True)
    if os.name == 'nt':
        content = content.replace(b'\r\n', b'\n')
    # with open('images/screen.png', 'wb') as f:
    #     f.write(content)
    # img_array = np.asarray(bytearray(content), dtype=np.uint8)
    return Image.open(BytesIO(content))


def save_collect_img(item_id, img):
    if not os.path.exists(collect_path + item_id):
        os.mkdir(collect_path + item_id)
    cv2.imwrite(collect_path + item_id + '/%s.png' % int(time.time() * 1000), img)


def prepare_train_resource():
    model = load_model()
    screen = screenshot()
    items = inventory.get_all_item_img_in_screen(screen)
    roi_list = []
    for x in items:
        roi = x['rectangle2'].copy()
        # inventory.show_img(roi)
        roi = np.transpose(roi, (2, 0, 1))
        roi_list.append(roi)
    res = predict(model, roi_list)
    print(res)
    for i in range(len(res[0])):
        item_id = res[0][i]
        print(res[1][i], inventory.item_map[int(item_id)])
        if res[1][i] < 0.1:
            item_id = 'other'
        else:
            keycode = inventory.show_img(items[i]['rectangle2'])
            if keycode != 13:
                item_id = 'other'
        print(item_id)
        save_collect_img(item_id, items[i]['rectangle'])


def test_cv_onnx():
    net = cv2.dnn.readNetFromONNX('ark_material.onnx')
    # screen = Image.open('images/screen.png')
    screen = screenshot()
    items = inventory.get_all_item_img_in_screen(screen)
    roi_list = []
    for x in items:
        roi = x['rectangle2'].copy()
        # inventory.show_img(roi)

        roi_list.append(roi)
        blob = cv2.dnn.blobFromImage(roi)
        net.setInput(blob)
        out = net.forward()

        # Get a class with a highest score.
        out = out.flatten()
        print(out)
        classId = np.argmax(out)
        confidence = out[classId]
        item_id = idx2id[classId]
        print(confidence, inventory.item_map[int(item_id)])
        inventory.show_img(roi)



if __name__ == '__main__':
    # train()
    test()
    # prepare_train_resource()
    # test_cv_onnx()



