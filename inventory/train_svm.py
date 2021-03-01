import cv2
import os
import inventory
import numpy as np


model_file = 'item_svm.dat'


def load_items():
    samples = []
    labels = []
    for file in os.listdir('images/item/'):
        a = file.split('.')
        if a[0].isdigit():
            item_id = int(a[0])
            print(item_id)
            item = cv2.imread('images/item/' + file)
            # item = cv2.imread('images/item/30014.png')
            # item = cv2.imread('images/item/30024.png')
            item_gray = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
            circles = inventory.get_circles(item_gray, 75, 100)
            img = inventory.crop_item_img(item, item_gray, circles[0])
            # inventory.show_img(img)
            # break
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # inventory.show_img(gray_img)
            # break
            feature = inventory.get_img_feature(gray_img)
            for _ in range(10):
                samples.append(feature)
                labels.append(item_id)
    samples = np.float32(samples)
    labels = np.array(labels)

    rand = np.random.RandomState(100)
    shuffle = rand.permutation(len(samples))
    samples = samples[shuffle]
    labels = labels[shuffle]
    print('samples: %s, labels: %s' % (len(samples), len(labels)))
    return samples, labels


def train():
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)

    svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.setC(0.1)

    # svm.setKernel(cv2.ml.SVM_RBF)  # cv2.ml.SVM_LINEAR
    # # svm.setDegree(0.0)
    # svm.setGamma(5.383)
    # # svm.setCoef0(0.0)
    svm.setC(0.01)

    samples, labels = load_items()

    svm.train(samples, cv2.ml.ROW_SAMPLE, labels)
    svm.save(model_file)


def test():
    from PIL import Image
    pil_img = Image.open('images/screen.png')
    item_images = inventory.get_all_item_img_in_screen(pil_img)
    for item_img in item_images:
        item = inventory.predict(item_img['circle'])
        print(item)
        inventory.show_img(item_img['circle'])


if __name__ == '__main__':
    # train()
    test()
