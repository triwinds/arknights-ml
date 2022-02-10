import subprocess
import cv2
import json
import numpy as np
from ppocronnx.predict_system import TextSystem


item_circle_radius = 64
itemreco_box_size = 142
half_box = itemreco_box_size // 2
ppocr = TextSystem(unclip_ratio=1.4, box_thresh=0.1, use_angle_cls=False)
ppocr.set_char_whitelist('.0123456789万')


def load_net_data():
    with open('ark_material.onnx', 'rb') as f:
        data = f.read()
        net = cv2.dnn.readNetFromONNX(data)
    with open('index_itemid_relation.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return net, data['idx2id'], data['id2idx'], data['idx2name'], data['idx2type']


net, idx2id, id2idx, idx2name, idx2type = load_net_data()


def get_circles(gray_img, min_radius=55, max_radius=65):
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 100, param1=50,
                               param2=30, minRadius=min_radius, maxRadius=max_radius)
    return circles[0]


def crop_item_middle_img(cv_item_img):
    # radius 60
    img_h, img_w = cv_item_img.shape[:2]
    ox, oy = img_w // 2, img_h // 2
    y1 = int(oy - 40)
    y2 = int(oy + 20)
    x1 = int(ox - 30)
    x2 = int(ox + 30)
    return cv_item_img[y1:y2, x1:x2]


def show_img(cv_img):
    cv2.imshow('test', cv_img)
    return cv2.waitKey(0)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_item_img(cv_screen, dbg_screen, center_x, center_y, ratio):
    img_h, img_w = cv_screen.shape[:2]
    x, y = int(center_x - half_box), int(center_y - half_box)
    if x < 0 or x + itemreco_box_size > img_w:
        return None
    cv_item_img = cv_screen[y:y + itemreco_box_size, x:x + itemreco_box_size]

    cv2.rectangle(dbg_screen, (x, y), (x + itemreco_box_size, y + itemreco_box_size), (255, 0, 0), 2)
    return {'item_img': cv_item_img,
            'item_pos': (int((x + itemreco_box_size // 2)/ratio), int((y + itemreco_box_size // 2)/ratio))}


def get_all_item_img_in_screen(cv_screen):
    h, w = cv_screen.shape[:2]
    ratio = 720 / h
    if h != 720:
        cv_screen = cv2.resize(cv_screen, (int(w * ratio), int(h * ratio)))
    gray_screen = cv2.cvtColor(cv_screen, cv2.COLOR_BGR2GRAY)
    dbg_screen = cv_screen.copy()
    # cv2.HoughCircles seems works fine for now
    circles: np.ndarray = get_circles(gray_screen)
    if circles is None:
        return []
    res = []
    for center_x, center_y, r in circles:
        cv2.circle(dbg_screen, (int(center_x), int(center_y)), int(r), (0, 0, 255), 2)
        item_img = get_item_img(cv_screen, dbg_screen, center_x, center_y, ratio)
        if item_img:
            res.append(item_img)
    # show_img(dbg_screen)
    # cv2.imwrite('demo-dbg.png', dbg_screen)
    return res


def get_item_info(cv_img, box_size=137):
    cv_img = cv2.resize(cv_img, (box_size, box_size))
    mid_img = crop_item_middle_img(cv_img)
    blob = cv2.dnn.blobFromImage(mid_img)
    net.setInput(blob)
    out = net.forward()

    # Get a class with a highest score.
    out = out.flatten()
    probs = softmax(out)
    classId = np.argmax(out)
    return probs[classId], idx2id[classId], idx2name[classId], idx2type[classId]


def get_quantity_ppocr(ori_img):
    img_h, img_w = ori_img.shape[:2]
    half_img = ori_img[int(img_h*0.65):img_h, 0:img_w]
    # show_img(half_img)
    res = ppocr.detect_and_ocr(half_img, 0.05)
    res = sorted(res, key=lambda x: x.score, reverse=True)
    if res:
        for ocr_res in res:
            numtext = ocr_res.ocr_text
            if numtext.isdigit():
                return int(numtext)


def screenshot():
    content = subprocess.check_output('adb exec-out "screencap -p"', shell=True)
    img_array = np.asarray(bytearray(content), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def main(with_quantity=True):
    screen = cv2.imread('demo.png')
    # screen = screenshot()
    # cv2.imwrite('demo.png', screen)
    item_images = get_all_item_img_in_screen(screen)
    for item_img in item_images:
        # prob 识别结果置信度
        # item_id, item_name, item_type 见 Kengxxiao/ArknightsGameData 的解包数据
        # https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData/master/zh_CN/gamedata/excel/item_table.json
        prob, item_id, item_name, item_type = get_item_info(item_img['item_img'])
        quantity = None
        if with_quantity:
            quantity = get_quantity_ppocr(item_img['item_img'])
        # name: 中级作战记录, quantity: 8416, pos: (235, 190), prob: 0.9656420946121216
        print(f"name: {item_name}, quantity: {quantity}, pos: {item_img['item_pos']}, prob: {prob}")
        # show_img(item_img['item_img'])


if __name__ == '__main__':
    # main(with_quantity=True)
    main(with_quantity=False)
