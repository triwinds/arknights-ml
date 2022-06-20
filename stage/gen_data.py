import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import cv_svm_ocr
import os

from stage import demo

target_dir = 'images/chars2'


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
        char_dir = f'{target_dir}/{c}'
        if not os.path.exists(char_dir):
            os.mkdir(char_dir)
        cv2.imwrite(char_dir + f'/{img_prefix}_%s.png' % c, char_img)


def add_test_char(test_char_dir):
    img_paths = os.listdir(test_char_dir)
    for img_name in img_paths:
        img = cv2.imread(f'{test_char_dir}/{img_name}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        real_tag_str = img_name[:-4]
        noise_size = None if not real_tag_str.isdigit() \
                             and 'EPISODE' not in real_tag_str else 1
        char_imgs = demo.crop_char_img(img, noise_size)
        if len(char_imgs) != len(real_tag_str):
            print(f'wrong crop: {img_name}, len: {len(char_imgs)}, real: {len(real_tag_str)}')
            continue
        for i, char_img in enumerate(char_imgs):
            cv2.imwrite(f'{target_dir}/{real_tag_str[i]}/{i}-{img_name}', ~char_img)


if __name__ == '__main__':
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    gen_data('Novecento WideBold.otf', 'gen_nw')
    gen_data('Bender.otf', 'gen_b', '0123456789')
    add_test_char('images/test')
    # gen_data('Novecento WideMedium.otf', 'gen_nwm')

