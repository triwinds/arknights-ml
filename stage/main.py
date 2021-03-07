import os
import subprocess
import time

import cv2
import numpy as np

import cv_svm_ocr

img_cache = {}
screenshot_cache = None
screenshot_rgb_cache = None
screenshot_time = None


def screenshot():
    global screenshot_cache
    global screenshot_time
    global screenshot_rgb_cache
    st = time.time()
    content = subprocess.check_output('adb exec-out "screencap -p"', shell=True)
    if os.name == 'nt':
        content = content.replace(b'\r\n', b'\n')
    # with open('images/screen.png', 'wb') as f:
    #     f.write(content)
    img_array = np.asarray(bytearray(content), dtype=np.uint8)
    screenshot_cache = cv2.imdecode(img_array, 0)
    screenshot_rgb_cache = cv2.imdecode(img_array, 1)
    screenshot_time = time.time() - st
    screenshot_cache = _resize_img(screenshot_cache)
    screenshot_rgb_cache = _resize_img(screenshot_rgb_cache)


def resize_img(img_path):
    global img_cache
    img1 = img_cache.get(img_path, cv2.imread(img_path, 0))
    img_cache[img_path] = img1
    img2 = screenshot_cache
    height, width = img1.shape[:2]
    ratio = 1080 / img2.shape[0]
    size = (int(width / ratio), int(height / ratio))
    return cv2.resize(img1, size, interpolation=cv2.INTER_AREA)


def _resize_img(img):
    height, width = img.shape[:2]
    if height == 1080:
        return img
    ratio = 1080 / height
    size = (int(width * ratio), int(height * ratio))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def image_to_position(image, m=0):
    global img_cache
    image_path = 'images/' + str(image) + '.png'
    screen = screenshot_cache
    template = resize_img(image_path)
    get_position(screen, template)


def thresholding(image):
    img = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if img[0, 0] < 127:
        img = ~img
    return img


def resize_cv_img(img, ratio, interpolation=cv2.INTER_CUBIC):
    return cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)


def get_position(screen, template):
    methods = [cv2.TM_CCOEFF_NORMED, cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED]
    result = cv2.matchTemplate(screen, template, methods[0])
    threshold = 0.8
    loc = np.where(result >= threshold)
    h, w = template.shape[:2]
    img_h, img_w = screen.shape[:2]
    tag_set = set()
    for pt in zip(*loc[::-1]):
        pos_key = '%d-%d' % (pt[0] / 100, pt[1] / 100)
        if pos_key in tag_set:
            continue
        tag_set.add(pos_key)
        # cv2.rectangle(screen, pt, (pt[0] + w, pt[1] + h), (7, 249, 151), 3)
        tag_w = 130
        if pt[0] + w + tag_w < img_w:
            tag = cut_tag(screen, w, pt)
            tag = thresholding(tag)
            remove_holes(tag)
            # cv2.imwrite('images/tmp/%s.png' % pos_key, tag)
            tag_str = cv_svm_ocr.do_ocr(tag)
            print(pos_key, tag_str)



def lock_screen():
    os.system('adb shell input keyevent 26')


def save_screenshot():
    global screenshot_rgb_cache
    screenshot()
    cv2.imwrite('images/screen.png', screenshot_rgb_cache)


def cut_tag(screen, w, pt):
    img_h, img_w = screen.shape[:2]
    tag_w = 130
    tag = thresholding(screen[pt[1] - 1:pt[1] + 40, pt[0] + w + 3:pt[0] + tag_w + w])
    # 130 像素不一定能将 tag 截全，所以再检查一次看是否需要拓宽 tag 长度
    for i in range(3):
        for j in range(40):
            if tag[j][tag_w - 4 - i] < 127:
                tag_w = 160
                if pt[0] + w + tag_w >= img_w:
                    return None
                tag = thresholding(screen[pt[1] - 1:pt[1] + 40, pt[0] + w + 3:pt[0] + tag_w + w])
                break
    return tag


def remove_holes(img):
    # 去除小连通域
    # findContours 只能处理黑底白字的图像, 所以需要进行一下翻转
    contours, hierarchy = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        # 计算区块面积
        area = cv2.contourArea(contours[i])
        if area < 8:
            # 将面积较小的点涂成白色，以去除噪点
            cv2.drawContours(img, [contours[i]], 0, 255, -1)


def prepare_train_resource(image_name, skip_save=False):
    global screenshot_cache
    image_path = 'images/' + str(image_name) + '.png'
    screen = screenshot_cache
    template = resize_img(image_path)
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(result >= threshold)
    h, w = template.shape[:2]
    img_h, img_w = screen.shape[:2]
    tag_set = set()
    for pt in zip(*loc[::-1]):
        pos_key = '%d-%d' % (pt[0] / 100, pt[1] / 100)
        if pos_key in tag_set:
            continue
        tag_set.add(pos_key)
        # cv2.rectangle(screen, pt, (pt[0] + w, pt[1] + h), (7, 249, 151), 3)
        tag_w = 130
        # 检查边缘像素是否超出截图的范围
        if pt[0] + w + tag_w < img_w:
            tag = cut_tag(screen, w, pt)
            tag = thresholding(tag)
            remove_holes(tag)
            cv2.imwrite('images/tmp/%s.png' % pos_key, tag)
            tag_str = cv_svm_ocr.do_ocr(tag)
            print(pos_key, tag_str)

            if not skip_save:
                char_imgs = cv_svm_ocr.crop_char_img(tag)
                for i in range(len(char_imgs)):
                    char_img = char_imgs[i]
                    c = cv_svm_ocr.predict(char_img, model_file='svm_data.dat')

                    char_dir = 'images/chars2/%s' % c
                    # char_dir = 'images/tmp'
                    if not os.path.exists(char_dir):
                        os.mkdir(char_dir)
                    cv2.imwrite(char_dir + '/%s.png' % int(time.time()*1000), char_img)


def get_train_resource(skip_save=False):

    while True:
        print('screenshot')
        screenshot()
        # load_screenshot_from_file()
        prepare_train_resource('stage_icon1', skip_save)
        prepare_train_resource('stage_icon2', skip_save)
        s = input('continue?')
        if s == 'n':
            break


def load_screenshot_from_file():
    global screenshot_cache
    screenshot_cache = cv2.imread('images/screen.png', cv2.IMREAD_GRAYSCALE)


def move_to_char2():
    img_dir = "images/old_chars"
    char2_dir = "images/chars2"
    for train_char in os.listdir(img_dir):
        print('train [%s]' % train_char)
        img_len = len(os.listdir(img_dir + '/' + train_char))
        print('load %s images' % img_len)
        for file_name in os.listdir(img_dir + '/' + train_char):
            img = cv2.imread(img_dir + '/%s/' % train_char + file_name, 0)
            remove_holes(img)
            min_y = None
            max_y = None
            h, w = img.shape[:2]
            for y1 in range(0, h):
                has_black = False
                for x1 in range(0, w):
                    if img[y1][x1] < 127:
                        has_black = True
                        if min_y is None:
                            min_y = y1
                        break
                if not has_black and min_y is not None and max_y is None:
                    max_y = y1
                    break
            if not os.path.exists(char2_dir + '/%s/' % train_char):
                os.mkdir(char2_dir + '/%s/' % train_char)
            img2 = img[min_y:max_y, 0:w]
            cv2.imwrite(char2_dir + '/%s/' % train_char + file_name, img2)


if __name__ == '__main__':
    # with open('images/screen.png', 'rb') as f:
    #     content = f.read()
    # img_array = np.asarray(bytearray(content), dtype=np.uint8)
    # screenshot_cache = cv2.imdecode(img_array, 0)

    # os.system('adb connect 192.168.3.68:5555')
    # os.system('adb kill-server')
    # while True:
    #     print('screenshot')
    #     screenshot()
    #     image_to_position('stage_icon1')
    #     image_to_position('stage_icon2')
    #     s = input('continue?')
    #     if s == 'n':
    #         break

    # screenshot()
    # save_screenshot()

    # img = cv2.imread('images/battle_start.png', 1)
    # img = resize_cv_img(img, 2/3, cv2.INTER_AREA)
    # cv2.imwrite('images/battle_start2.png', img)

    # save_screenshot()

    get_train_resource(True)

    # move_to_char2()
    # print(os.listdir('images/chars2'))

