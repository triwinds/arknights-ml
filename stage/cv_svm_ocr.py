import os
import io
import random
import zipfile
from functools import lru_cache

import cv2
import numpy as np


def hog(img):
    bin_n = 16  # Number of bins
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(8 * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:16, :16], bins[16:, :16], bins[:16, 16:], bins[16:, 16:]
    mag_cells = mag[:16, :16], mag[16:, :16], mag[:16, 16:], mag[16:, 16:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  # hist is a 64 bit vector
    return hist


def gamma_trans(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def train(img_dir='images/chars', output_file='svm_data.dat'):
    svm = cv2.ml.SVM_create()
    # svm.setType(cv2.ml.SVM_C_SVC)
    # svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(0.01)

    samples, labels = load_train_resource(img_dir)

    svm.train(samples, cv2.ml.ROW_SAMPLE, labels)
    svm.save(output_file)
    with open(output_file, 'rb') as f:
        zf = zipfile.ZipFile('svm_data.zip', 'w', zipfile.ZIP_DEFLATED)
        zf.writestr('svm_data.dat', f.read())
    zf.close()


def load_train_resource(img_dir):
    samples = []
    labels = []
    print(img_dir)
    for train_char in os.listdir(img_dir):
        print('train [%s]' % train_char)
        img_len = len(os.listdir(img_dir + '/' + train_char))
        print('load %s images' % img_len)
        for file_name in os.listdir(img_dir + '/' + train_char):
            img = cv2.imread(img_dir + '/%s/' % train_char + file_name, 0)
            samples.append(get_img_feature(img))
            labels.append(ord(train_char))
        train_cells = samples

    samples = np.float32(samples)
    labels = np.array(labels)

    # rand = np.random.RandomState(321)
    # shuffle = rand.permutation(len(samples))
    # samples = samples[shuffle]
    # labels = labels[shuffle]
    print('samples: %s, labels: %s' % (len(samples), len(labels)))
    return samples, labels


def predict(gray_img, model_file='svm_data1.dat'):
    svm = load_svm(model_file)
    res = svm.predict(np.float32([get_img_feature(gray_img)]))
    return chr(int(res[1][0][0]))


def get_img_feature(img):
    # svm 中针对固定字体的图像, 直接 resize 图像作为特征比 hog 效果更好
    return resize_char(img).reshape(256, 1)


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


@lru_cache(2)
def load_svm(model_file='svm_data1.dat'):
    if model_file.endswith('.dat'):
        return cv2.ml.SVM_load(model_file)
    else:
        return load_svm_from_zip(model_file)


def load_svm_from_zip(model_file='svm_data.zip'):
    with open(model_file, 'rb') as f:
        # bio = io.BytesIO(f.read())
        zf = zipfile.ZipFile(f, 'r')
        ydoc = zf.read('svm_data.dat').decode('utf-8')
        fs = cv2.FileStorage(ydoc, cv2.FileStorage_READ | cv2.FileStorage_MEMORY)
        svm = cv2.ml.SVM_create()
        svm.read(fs.getFirstTopLevelNode())
        assert svm.isTrained()
        return svm


def crop_char_img_old(img):
    h, w = img.shape[:2]
    has_black = False
    last_x = None
    res = []
    for x in range(0, w):
        for y in range(0, h):
            has_black = False
            if img[y][x] < 127:
                has_black = True
                if not last_x:
                    last_x = x
                break
        if not has_black and last_x:
            if x - last_x > 5:
                res.append(img[0:h, last_x:x])
            last_x = None
    return res


def crop_char_img(img):
    h, w = img.shape[:2]
    has_black = False
    last_x = None
    res = []
    for x in range(0, w):
        for y in range(0, h):
            has_black = False
            if img[y][x] < 127:
                has_black = True
                if not last_x:
                    last_x = x
                break
        if not has_black and last_x:
            if x - last_x >= 3:
                min_y = None
                max_y = None
                for y1 in range(0, h):
                    has_black = False
                    for x1 in range(last_x, x):
                        if img[y1][x1] < 127:
                            has_black = True
                            if min_y is None:
                                min_y = y1
                            break
                    if not has_black and min_y is not None and max_y is None:
                        max_y = y1
                        break
                res.append(img[min_y:max_y, last_x:x])
            last_x = None
    return res


def threshold_cv_img(image):
    return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def pil_to_cv_gray_img(pil_img):
    arr = np.asarray(pil_img, dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def invert_cv_gray_img_color(img):
    return ~img


def recognize_stage_tags(img, template, prefix_len=2):
    screen = pil_to_cv_gray_img(img.copy())
    template = pil_to_cv_gray_img(template.copy())
    result = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(result >= threshold)
    h, w = template.shape[:2]
    img_h, img_w = screen.shape[:2]
    tag_set = set()
    res = []
    for pt in zip(*loc[::-1]):
        pos_key = '%d-%d' % (pt[0] / 100, pt[1] / 100)
        if pos_key in tag_set:
            continue
        tag_set.add(pos_key)
        tag_w = 120 + 10 * prefix_len
        if pt[0] + w + tag_w < img_w:
            tag = screen[pt[1] - 1:pt[1] + 40, pt[0] + w + 3:pt[0] + tag_w + w]
            if tag[0, 0] < 127:
                tag = invert_cv_gray_img_color(tag)
            tag = threshold_cv_img(tag)
            # cv.imshow('tag', tag)
            # cv.waitKey(0)
            res.append({'tag_img': tag, 'pos': (pt[0] + (tag_w / 2), pt[1] + 20)})
    return res


def do_ocr(img):
    char_imgs = crop_char_img(img)
    s = ''
    for char_img in char_imgs:
        c = predict(char_img, model_file='svm_data.dat')
        s += c
    return s


def check(img_dir='images/old_chars', model_file='svm_data1.dat'):
    total = 0
    correct = 0
    for test_char in os.listdir(img_dir):
        for _ in range(20):
            total += 1
            file_name = random.choice(os.listdir(img_dir + '/' + test_char))
            img = cv2.imread(img_dir + '/%s/' % test_char + file_name, 0)
            c = predict(img, model_file)
            if c == test_char:
                correct += 1
            else:
                print(c, test_char)
    print('%s/%s = %s' % (correct, total, correct / total))


if __name__ == '__main__':
    # train('images/old_chars', 'svm_data1.dat')
    #
    # svm = cv2.ml.SVM_load('svm_data1.dat')
    # img = cv2.imread('images/old_chars/6/1590847037680.png', 0)
    # print(predict(img))
    # check('images/old_chars', 'svm_data1.dat')

    # train_KNearest()
    # check_KNearest()

    train('images/chars2', 'svm_data.dat')
    check('images/chars2', 'svm_data.zip')

    # train_ann()
