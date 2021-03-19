import json
import os
import subprocess
import time
from io import BytesIO

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import inventory
from focal_loss import FocalLoss

collect_path = 'images/collect/'


def update_resources():
    from dl_data import download_icons
    print('更新素材')
    download_icons()


update_resources()


def dump_index_itemid_relation():
    dump_data = {
        'idx2id': [],
        'id2idx': {}
    }
    index = 0
    for dirpath in os.listdir(collect_path):
        item_id = dirpath
        dump_data['idx2id'].append(item_id)
        dump_data['id2idx'][item_id] = index
        index += 1
    with open('index_itemid_relation.json', 'w') as f:
        json.dump(dump_data, f)
    return dump_data['idx2id'], dump_data['id2idx']


def load_images():
    img_map = {}
    gray_img_map = {}
    item_id_map = {}
    circle_map = {}
    img_files = []
    collect_list = os.listdir('images/collect')
    collect_list.sort()
    weights = []
    for cdir in collect_list:
        dirpath = 'images/collect/' + cdir
        sub_dir_files = os.listdir(dirpath)
        weights.append(len(sub_dir_files))
        for filename in sub_dir_files:
            filepath = os.path.join(dirpath, filename)
            with open(filepath, 'rb') as f:
                nparr = np.frombuffer(f.read(), np.uint8)
                # convert to image array
                image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                image = cv2.resize(image, (128, 128))
                if image.shape[-1] == 4:
                    image = image[..., :-1]
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                img_map[filepath] = image
                gray_img_map[filepath] = gray_img
                img_files.append(filepath)
                item_id_map[filepath] = cdir
                circles = inventory.get_circles(gray_img, 50, 100)
                circle_map[filepath] = circles[0]
    weights_t = 1 / torch.as_tensor(weights)
    return img_map, gray_img_map, img_files, item_id_map, circle_map, weights_t


idx2id, id2idx = dump_index_itemid_relation()
img_map, gray_img_map, img_files, item_id_map, circle_map, weights_t = load_images()
NUM_CLASS = len(idx2id)
print('NUM_CLASS', NUM_CLASS)


def crop_item_middle_img(cv_item_img, ox, oy, radius):
    ratio = radius / 60
    y1 = int(oy - (40 * ratio))
    y2 = int(oy + (24 * ratio))
    x1 = int(ox - (32 * ratio))
    x2 = int(ox + (32 * ratio))
    return cv2.resize(cv_item_img[y1:y2, x1:x2], (64, 64))


def get_noise_data():
    images_np = np.random.rand(40, 64, 64, 3)
    labels_np = np.asarray(['other']).repeat(40)
    return images_np, labels_np


def get_data():
    images = []
    labels = []
    for filepath in img_files:
        item_id = item_id_map[filepath]

        # if not item_id.isdigit() and np.random.randint(0, 100) >= 30:
        #     continue

        image = img_map[filepath]
        # print(filepath)
        # inventory.show_img(image)
        c = circle_map[filepath]
        t = 30 if item_id.isdigit() else 1
        for _ in range(t):
            ox = c[0] + np.random.randint(-5, 5)
            oy = c[1] + np.random.randint(-5, 5)
            img = crop_item_middle_img(image, ox, oy, c[2])

            image_aug = img

            images.append(image_aug)
            labels.append(id2idx[item_id])
    images_np = np.transpose(np.stack(images, 0), [0, 3, 1, 2])
    labels_np = np.array(labels)

    rand = np.random.RandomState(321)
    shuffle = rand.permutation(len(images_np))
    images_np = images_np[shuffle]
    labels_np = labels_np[shuffle]

    # print(images_np.shape)
    return images_np, labels_np


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 10, 5, stride=2, padding=2),  # 10 * 32 * 32
            nn.ReLU(True),
            nn.MaxPool2d(4, 4))  # 10 * 8 * 8

        self.fc = nn.Sequential(
            nn.Linear(640, 320),
            nn.ReLU(True),
            nn.Linear(320, 120),
            nn.ReLU(True),
            nn.Linear(120, NUM_CLASS))

    def forward(self, x):
        out = self.conv(x)
        out = out.reshape(-1, 640)
        out = self.fc(out)
        return out


focalloss = FocalLoss(NUM_CLASS, alpha=weights_t)
# BCEWithLogitsLoss = nn.BCEWithLogitsLoss(weights_t)


def compute_loss(x, label):
    loss = focalloss(x, label)
    prec = (x.argmax(1) == label).float().mean()
    return loss, prec


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('train on:', device)
    model = Cnn().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    step = 0
    prec = 0
    target_step = 300
    while step < target_step or prec < 1 or step > 2*target_step:
        images_aug_np, label_np = get_data()
        images_aug = torch.from_numpy(images_aug_np).float().to(device)
        label = torch.from_numpy(label_np).long().to(device)
        optim.zero_grad()
        score = model(images_aug)
        loss, prec = compute_loss(score, label)
        loss.backward()
        optim.step()
        if step < 10 or step % 10 == 0:
            print(step, loss.item(), prec.item())
        step += 1
    torch.save(model.state_dict(), './model.bin')
    torch.onnx.export(model, images_aug, 'ark_material.onnx')


def load_model():
    model = Cnn()
    device = torch.device('cpu')
    model.load_state_dict(torch.load('./model.bin', map_location=device))
    model.eval()
    return model


def predict(model, roi_list):
    """
    Image size of 720p is recommended.
    """
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
    screen = Image.open('images/screen.png')
    # screen = screenshot()
    items = inventory.get_all_item_img_in_screen(screen)
    roi_list = []
    for x in items:
        roi = x['rectangle2'].copy()
        # roi = roi / 255
        roi = np.transpose(roi, (2, 0, 1))
        roi_list.append(roi)
    res = predict(model, roi_list)
    print(res)
    for i in range(len(res[0])):
        item_id = res[0][i]
        if item_id == 'other':
            print(res[1][i], 'other')
        else:
            print(res[1][i], inventory.item_map[item_id])
        inventory.show_img(items[i]['rectangle'])


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
        # roi = roi / 255
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
    screen = Image.open('images/screen.png')
    # screen = screenshot()
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
        print(confidence, inventory.item_map[item_id] if item_id.isdigit() else item_id)
        inventory.show_img(x['rectangle'])


def export_onnx():
    model = load_model()
    screen = Image.open('images/screen.png')
    items = inventory.get_all_item_img_in_screen(screen)
    roi_list = []
    for x in items:
        roi = x['rectangle2'].copy()
        roi = np.transpose(roi, (2, 0, 1))
        roi_list.append(roi)
    roi_np = np.stack(roi_list, 0)
    roi_t = torch.from_numpy(roi_np).float()
    torch.onnx.export(model, roi_t, 'ark_material.onnx')


def test_single_img(img_path=''):
    model = load_model()
    image = img_map[img_path]
    roi_list = []
    roi = inventory.crop_item_middle_img(image, 60)
    roi = np.transpose(roi, (2, 0, 1))
    # inventory.show_img(roi)
    roi_list.append(roi)
    res = predict(model, roi_list)
    item_id = res[0][0]
    if item_id == 'other':
        print(res[1][0], 'other')
    else:
        print(res[1][0], inventory.item_map[item_id])
    inventory.show_img(image)


if __name__ == '__main__':
    train()
    # test()
    # prepare_train_resource()
    # export_onnx()
    # test_cv_onnx()
    # print(cv2.getBuildInformation())
    # test_single_img('images/collect/other\\罗德岛物资配给证书.png')

