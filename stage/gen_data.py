import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import cv_svm_ocr
import os


if __name__ == '__main__':
    img = np.ones((40, 50, 3), np.uint8)
    b, g, r, a = 0, 0, 0, 0

    fontpath = "./Mada-Medium.otf"
    font = ImageFont.truetype(fontpath, 45)
    chars = '-0123456789QWERTYUIOPASDFGHJKLZXCVBNM'
    for c in chars:
        img = np.zeros((40, 40, 3), np.uint8)
        img += 255
        img_pil = Image.fromarray(img.copy())
        draw = ImageDraw.Draw(img_pil)
        draw.text((3, -10), c, font=font, fill=(b, g, r, a))
        img = np.array(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv_svm_ocr.threshold_cv_img(img)
        char_imgs = cv_svm_ocr.crop_char_img(img)
        print(c, len(char_imgs))
        assert len(char_imgs) == 1
        char_img = char_imgs[0]
        char_dir = 'images/chars2/%s' % c
        if not os.path.exists(char_dir):
            os.mkdir(char_dir)
        cv2.imwrite(char_dir + '/gen_%s.png' % c, char_img)
