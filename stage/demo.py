import subprocess
import cv2
import numpy as np


idx2id = ['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def load_onnx_model():
    with open('chars.onnx', 'rb') as f:
        data = f.read()
        net = cv2.dnn.readNetFromONNX(data)
        return net


net = load_onnx_model()


def predict_cv(img, noise_size=None):
    char_imgs = crop_char_img(img, noise_size)
    if not char_imgs:
        return ''
    roi_list = [np.expand_dims(resize_char(x), 2) for x in char_imgs]
    blob = cv2.dnn.blobFromImages(roi_list)
    net.setInput(blob)
    scores = net.forward()
    predicts = scores.argmax(1)
    return ''.join([idx2id[p] for p in predicts])


def resize_char(img):
    h, w = img.shape[:2]
    scale = 16 / max(h, w)
    h = int(h * scale)
    w = int(w * scale)
    img2 = np.zeros((16, 16)).astype(np.uint8)
    img = cv2.resize(img, (w, h))
    img2[0:h, 0:w] = img
    # cv2.imshow('test', img2)
    # cv2.waitKey()
    return img2


def crop_char_img(img, noise_size=None, include_last_char=False):
    h, w = img.shape[:2]
    has_white = False
    last_x = None
    res = []
    if noise_size is None:
        noise_size = 3 if h > 40 else 2
    for x in range(0, w):
        for y in range(0, h - noise_size + 1):
            has_white = False
            flag = False
            if img[y][x] > 127:
                flag = True
                for i in range(noise_size):
                    if img[y+i][x] < 127:
                        flag = False
            if flag:
                has_white = True
                if not last_x:
                    last_x = x
                break
        if not has_white and last_x:
            if x - last_x >= noise_size // 2:
                x_char_img = img[:, last_x:x]
                res.append(crop_x_char(x_char_img))
            last_x = None
        if include_last_char and has_white and last_x != w - 1 and x == w - 1:
            x_char_img = img[:, last_x:w]
            res.append(crop_x_char(x_char_img))
    return res


def crop_x_char(x_char_img):
    min_y, max_y = 0, x_char_img.shape[0]
    y_max = np.max(x_char_img, axis=1)
    for i in range(len(y_max)):
        if y_max[i] >= 127:
            min_y = i
            break
    for i in range(len(y_max) - 1, 0, -1):
        if y_max[i] >= 127:
            max_y = i + 1
            break
    return x_char_img[min_y:max_y, :]

def thresholding(image):
    img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if img[0, 0] > 127:
        img = ~img
    return img


def cut_tag(screen, w, pt):
    img_h, img_w = screen.shape[:2]
    tag_w, tag_h = 130, 36
    tag = thresholding(screen[pt[1] - 1:pt[1] + tag_h, pt[0] + w + 3:pt[0] + tag_w + w])
    # 130 像素不一定能将 tag 截全，所以再检查一次看是否需要拓宽 tag 长度
    for i in range(3):
        for j in range(tag_h):
            if tag[j][tag_w - 4 - i] > 127:
                tag_w = 150
                if pt[0] + w + tag_w >= img_w:
                    return None
                tag = thresholding(screen[pt[1] - 1:pt[1] + tag_h, pt[0] + w + 3:pt[0] + tag_w + w])
                break
    return tag


def remove_holes(img):
    # 去除小连通域
    h, w = img.shape[:2]
    noise_size = 15 if h > 25 else 8
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        # 计算区块面积
        area = cv2.contourArea(contours[i])
        if area < noise_size:
            # 将面积较小的点涂成黑色，以去除噪点
            cv2.drawContours(img, [contours[i]], 0, 0, -1)


def recognize_stage_tags(cv_screen, template, ccoeff_threshold=0.75):
    screen = cv2.cvtColor(cv_screen, cv2.COLOR_BGR2GRAY)
    img_h, img_w = screen.shape[:2]
    ratio = 1080 / img_h
    if ratio != 1:
        ratio = 1080 / img_h
        screen = cv2.resize(screen, (int(img_w * ratio), 1080))
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= ccoeff_threshold)
    h, w = template.shape[:2]
    img_h, img_w = screen.shape[:2]
    tag_set = set()
    tag_set2 = set()
    res = []
    dbg_screen = None
    for pt in zip(*loc[::-1]):
        pos_key = (pt[0] // 100, pt[1] // 100)
        pos_key2 = (int(pt[0] / 100 + 0.5), int(pt[1] / 100 + 0.5))
        if pos_key in tag_set or pos_key2 in tag_set2:
            continue
        tag_set.add(pos_key)
        tag_set2.add(pos_key2)
        tag_w = 130
        # 检查边缘像素是否超出截图的范围
        if pt[0] + w + tag_w < img_w:
            tag = cut_tag(screen, w, pt)
            if tag is None:
                continue
            remove_holes(tag)
            tag_str = do_tag_ocr(tag, 3)
            if len(tag_str) < 3:
                if dbg_screen is None:
                    dbg_screen = screen.copy()
                cv2.rectangle(dbg_screen, pt, (pt[0] + w + tag_w, pt[1] + h), 0, 3)
                continue
            pos = (int((pt[0] + (tag_w / 2)) / ratio), int((pt[1] + 20) / ratio))
            res.append({'pos': pos, 'tag_str': tag_str})

    return res


def do_tag_ocr(img, noise_size=None):
    # 黑底白字
    return predict_cv(img, noise_size)


stage_icon1 = cv2.imread('images/stage_icon1.png', cv2.IMREAD_GRAYSCALE)
stage_icon2 = cv2.imread('images/stage_icon2.png', cv2.IMREAD_GRAYSCALE)
stage_icon_ex1 = cv2.imread('images/stage_icon_ex1.png', cv2.IMREAD_GRAYSCALE)
normal_icons = [stage_icon1, stage_icon2]
extra_icons = [stage_icon_ex1]


def recognize_all_screen_stage_tags(cv_screen, allow_extra_icons=False):
    tags_map = {}
    if allow_extra_icons:
        for icon in extra_icons:
            for tag in recognize_stage_tags(cv_screen, icon, 0.75):
                tags_map[tag['tag_str']] = tag['pos']
    for icon in normal_icons:
        for tag in recognize_stage_tags(cv_screen, icon):
            tags_map[tag['tag_str']] = tag['pos']
    return tags_map


def screenshot():
    content = subprocess.check_output('adb exec-out "screencap -p"', shell=True)
    img_array = np.asarray(bytearray(content), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


if __name__ == '__main__':
    stage_map = recognize_all_screen_stage_tags(cv2.imread('demo.png'))
    # stage_map = recognize_all_screen_stage_tags(screenshot())
    print(stage_map)
