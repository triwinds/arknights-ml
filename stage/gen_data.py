import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import cv_svm_ocr
import os


def gen_data(font_path, img_prefix, chars='-0123456789QWERTYUIOPASDFGHJKLZXCVBNM'):
    # img = np.ones((60, 60, 3), np.uint8)
    b, g, r, a = 0, 0, 0, 0

    font = ImageFont.truetype(font_path, 45)
    for c in chars:
        img = np.zeros((60, 60, 3), np.uint8)
        img += 255
        img_pil = Image.fromarray(img.copy())
        draw = ImageDraw.Draw(img_pil)
        draw.text((3, -10), c, font=font, fill=(b, g, r, a))
        img = np.array(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv_svm_ocr.threshold_cv_img(img)
        # cv2.imshow('test', img)
        # cv2.waitKey()
        char_imgs = cv_svm_ocr.crop_char_img(img)
        print(c, len(char_imgs))
        assert len(char_imgs) == 1
        char_img = char_imgs[0]
        char_dir = 'images/chars2/%s' % c
        if not os.path.exists(char_dir):
            os.mkdir(char_dir)
        cv2.imwrite(char_dir + f'/{img_prefix}_%s.png' % c, char_img)


if __name__ == '__main__':
    if not os.path.exists('images/chars2'):
        os.mkdir('images/chars2')
    gen_data('Novecento WideBold.otf', 'gen_nw')
    gen_data('Mada-Medium.otf', 'gen_mm')
    gen_data('Bender.otf', 'gen_b', '0123456789')
