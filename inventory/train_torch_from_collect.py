import json
import os
import subprocess
import time
# from functools import lru_cache
from io import BytesIO

import cv2
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
from PIL import Image

import inventory
from focal_loss import FocalLoss
from data_util import get_ignore_item_ids

collect_path = 'images/collect/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def update_resources(exit_if_not_update=False):
    from dl_data import download_icons
    print('更新素材')
    updated = download_icons()
    if not updated and exit_if_not_update:
        print('Nothing new, exit.')
        exit(0)


update_resources(True)


def dump_index_itemid_relation(exclude_collect):
    from dl_data import get_items_id_map
    items_id_map = get_items_id_map()
    dump_data = {
        'idx2id': [],
        'id2idx': {},
        'idx2name': [],
        'idx2type': [],
        'time': int(time.time() * 1000)
    }
    collect_list = os.listdir('images/collect')
    collect_list.sort()
    index = 0
    for dirpath in collect_list:
        if dirpath in exclude_collect:
            continue
        item_id = dirpath
        dump_data['idx2id'].append(item_id)
        dump_data['idx2name'].append(get_item_name(item_id, items_id_map))
        dump_data['idx2type'].append(get_item_type(item_id, items_id_map))
        dump_data['id2idx'][item_id] = index
        index += 1
    with open('index_itemid_relation.json', 'w', encoding='utf-8') as f:
        json.dump(dump_data, f, ensure_ascii=False)
    return dump_data['idx2id'], dump_data['id2idx'], dump_data['idx2name']


def get_item_name(item_id, items_id_map):
    item_name = '其它'
    if item_id.startswith('@'):
        item_name = get_manual_item_name(item_id)
    elif item_id != 'other':
        item_name = items_id_map[item_id]['name']
    return item_name


def get_manual_item_name(item_id):
    default_name = item_id
    tmp_map = {
        '@charm_r0_p9': '黄金筹码',
        '@charm_r0_p12': '错版硬币',
        '@charm_r0_p18': '双日城大乐透',
        '@charm_r1_p20': '标志物 - 20代金券',
        '@charm_r2_p40': '标志物 - 40代金券',
        '@charm_r3_p60': '沙兹专业镀膜装置',
        '@charm_r3_p150': '翡翠庭院至臻',
    }
    return tmp_map.get(item_id, default_name)


def get_item_type(item_id, items_id_map):
    item_type = 'other'
    if item_id.startswith('@'):
        item_type = 'manual_collect'
        if item_id.startswith('@charm'):
            item_type = 'special_report_item'
    elif item_id != 'other':
        item_type = items_id_map[item_id]['itemType']
    return item_type


def load_images():
    img_map = {}
    gray_img_map = {}
    item_id_map = {}
    circle_map = {}
    img_files = []
    collect_list = os.listdir('images/collect')
    collect_list.sort()
    weights = []
    empty_collect = []
    for cdir in collect_list:
        dirpath = 'images/collect/' + cdir
        sub_dir_files = os.listdir(dirpath)
        weight = 0
        for filename in sub_dir_files:
            filepath = os.path.join(dirpath, filename)
            with open(filepath, 'rb') as f:
                nparr = np.frombuffer(f.read(), np.uint8)
                # convert to image array
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image = cv2.resize(image, (140, 140))
                gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                circles = inventory.get_circles(gray_img, 50, 100)
                if circles is None:
                    print(f'No circle found in {filepath}, skip.')
                    continue
                circle_map[filepath] = circles[0]
                img_map[filepath] = torch.from_numpy(np.transpose(image, (2, 0, 1)))\
                    .float().to(device)
                gray_img_map[filepath] = gray_img
                img_files.append(filepath)
                item_id_map[filepath] = cdir
                weight += 1
        if weight == 0:
            print(f'empty_collect {cdir}')
            empty_collect.append(cdir)
            continue
        weights.append(weight)
    weights_t = torch.as_tensor(weights)
    weights_t[weights_t > 80] = 80
    weights_t = 1 / weights_t
    return img_map, gray_img_map, img_files, item_id_map, circle_map, weights_t, empty_collect


img_map, gray_img_map, img_files, item_id_map, circle_map, weights_t, empty_collect = load_images()
idx2id, id2idx, idx2name = dump_index_itemid_relation(empty_collect)
NUM_CLASS = len(idx2id)
print('NUM_CLASS', NUM_CLASS)


def crop_item_middle_img(cv_item_img, ox, oy):
    # ratio = radius / 60
    ratio = 1
    y1 = int(oy - (40 * ratio))
    y2 = int(oy + (20 * ratio))
    x1 = int(ox - (30 * ratio))
    x2 = int(ox + (30 * ratio))
    # return cv2.resize(cv_item_img[y1:y2, x1:x2], (64, 64))
    return cv_item_img[y1:y2, x1:x2]


def crop_tensor_middle_img(cv_item_img, ox, oy, radius):
    # ratio = radius / 60
    ratio = 1
    y1 = int(oy - (40 * ratio))
    y2 = int(oy + (20 * ratio))
    x1 = int(ox - (30 * ratio))
    x2 = int(ox + (30 * ratio))
    img_t = cv_item_img[..., y1:y2, x1:x2]
    # return F.interpolate(img_t, size=64, mode='bilinear')
    return img_t


def get_noise_data():
    images_np = np.random.rand(40, 64, 64, 3)
    labels_np = np.asarray(['other']).repeat(40)
    return images_np, labels_np


max_resize_ratio = 100


IGNORE_ITEM_ID = get_ignore_item_ids()
print('IGNORE_ITEM_ID', IGNORE_ITEM_ID)


def get_data(img_files, item_id_map, circle_map, img_map):
    images = []
    labels = []
    for filepath in img_files:
        item_id = item_id_map[filepath]
        if item_id in IGNORE_ITEM_ID:
            continue
        c = circle_map[filepath]
        t = 4 if item_id != 'other' else 1
        for _ in range(t):
            ox = c[0] + np.random.randint(-5, 5)
            oy = c[1] + np.random.randint(-5, 5)
            # ratio = np.random.randint(-max_resize_ratio, max_resize_ratio)
            # img_t = get_resized_img(filepath, ratio)
            img_t = img_map[filepath]
            img_t = crop_tensor_middle_img(img_t, ox, oy, c[2])
            image_aug = img_t

            images.append(image_aug)
            labels.append(id2idx[item_id])
    images_t = torch.stack(images)
    labels_t = torch.from_numpy(np.array(labels)).long().to(device)

    # print(images_np.shape)
    return images_t, labels_t


class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=3, padding=2),  # 32 * 20 * 20
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.AvgPool2d(5, 5),  # 32 * 4 * 4
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 32 * 2 * 2
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # nn.AvgPool2d(2, 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 2 * 2, 2 * NUM_CLASS),
            nn.ReLU(True),
            nn.Linear(2 * NUM_CLASS, NUM_CLASS))

    def forward(self, x):
        x /= 255.
        out = self.conv(x)
        out = out.reshape(-1, 32 * 2 * 2)
        out = self.fc(out)
        return out


def train():
    criterion = FocalLoss(NUM_CLASS, alpha=weights_t)
    criterion.to(device)

    def compute_loss(x, label):
        loss = criterion(x, label)
        prec = (x.argmax(1) == label).float().mean()
        return loss, prec

    print('train on:', device)
    model = Cnn().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    step = 0
    prec = 0
    target_step = 1200
    last_time = time.monotonic()
    is_saved = False
    best = 999
    while step < target_step or not is_saved:
        images_t, labels_t = get_data(img_files, item_id_map, circle_map, img_map)
        optim.zero_grad()
        score = model(images_t)
        loss, prec = compute_loss(score, labels_t)
        loss.backward()
        optim.step()
        if step < 10 or step % 50 == 0:
            print(step, loss.item(), prec.item(), time.monotonic() - last_time)
            last_time = time.monotonic()
        step += 1
        if step > target_step - 300 and best > loss.item():
            model.eval()
            if test(model):
                best = loss.item()
                print(f'save best {best}')
                model.train()
                torch.save(model.state_dict(), './model.pth')
                torch.onnx.export(model, torch.rand((1, 3, 60, 60)).to(device), 'ark_material.onnx')
                is_saved = True
            else:
                model.train()


    from dl_data import request_get
    request_get('https://purge.jsdelivr.net/gh/triwinds/arknights-ml@latest/inventory/index_itemid_relation.json', True)
    request_get('https://purge.jsdelivr.net/gh/triwinds/arknights-ml@latest/inventory/ark_material.onnx', True)
    request_get('https://purge.jsdelivr.net/gh/triwinds/arknights-ml@main/inventory/index_itemid_relation.json', True)
    request_get('https://purge.jsdelivr.net/gh/triwinds/arknights-ml@main/inventory/ark_material.onnx', True)


def load_model():
    model = Cnn()
    device = torch.device('cpu')
    model.load_state_dict(torch.load('./model.pth', map_location=device))
    model.eval()
    return model


def predict(model, roi_list):
    """
    Image size of 720p is recommended.
    """
    roi_np = np.stack(roi_list, 0)
    roi_t = torch.from_numpy(roi_np).float().to(device)
    with torch.no_grad():
        score = model(roi_t)
        probs = nn.Softmax(1)(score)
        predicts = score.argmax(1)

    probs = probs.cpu().data.numpy()
    predicts = predicts.cpu().data.numpy()
    return [(idx2id[idx], idx) for idx in predicts], [probs[i, predicts[i]] for i in range(len(roi_list))]


def test(model):
    # model = load_model()
    # screen = Image.open('images/screen.png')
    collect_list = os.listdir('images/collect')
    collect_list.sort()
    items = []
    for cdir in collect_list:
        dirpath = 'images/collect/' + cdir
        sub_dir_files = os.listdir(dirpath)
        filename = sub_dir_files[0]
        filepath = os.path.join(dirpath, filename)
        with open(filepath, 'rb') as f:
            nparr = np.frombuffer(f.read(), np.uint8)
            # convert to image array
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = crop_item_middle_img(cv2.resize(image, (140, 140)), 70, 70)
            items.append(image)
    roi_list = []
    for x in items:
        roi = x
        # roi = roi / 255
        roi = np.transpose(roi, (2, 0, 1))
        roi_list.append(roi)
    res = predict(model, roi_list)
    # print(res)
    for i in range(len(res[0])):
        item_id = res[0][i][0]
        expect_id = collect_list[i]
        # print(f"{item_id}/{expect_id}, {res[1][i]:.3f}")
        if item_id != expect_id and expect_id not in {'randomMaterial_1', 'randomMaterial_5'}:
            # inventory.show_img(items[i])
            print(f'Wrong predict: {item_id}/{expect_id}, {res[1][i]}')
            return False
    return True


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
    screen = inventory.screenshot()
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


def prepare_train_resource2():
    screen = inventory.screenshot()
    items = inventory.get_all_item_img_in_screen(screen, 2.15)
    for item in items:
        cv2.imwrite(f'images/manual_collect/{int(time.time() * 1000)}', item['rectangle'])


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def test_cv_onnx():
    net = cv2.dnn.readNetFromONNX('ark_material.onnx')
    screen = Image.open('images/screen.png')
    # screen = screenshot()
    items = inventory.get_all_item_img_in_screen(screen)
    for x in items:
        roi = x['rectangle2']
        # inventory.show_img(roi)
        blob = cv2.dnn.blobFromImage(roi)
        net.setInput(blob)
        out = net.forward()

        # Get a class with a highest score.
        out = out.flatten()
        out = softmax(out)
        # print(out)
        classId = np.argmax(out)
        # confidence = out[classId]
        confidence = out[classId]
        item_id = idx2id[classId]
        print(confidence, inventory.item_map[item_id] if item_id.isdigit() else item_id)
        # inventory.show_img(x['rectangle'])


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


if __name__ == '__main__':
    train()
    # prepare_train_resource()
    # prepare_train_resource2()
    # export_onnx()
    # test_cv_onnx()
    # print(cv2.getBuildInformation())
