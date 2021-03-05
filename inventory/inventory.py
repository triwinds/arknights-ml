import numpy as np
import cv2
import json



def load_item_map():
    with open('penguin_cache.json', 'r') as f:
        penguin_cache = json.load(f)
        item_map = {}
        for item in penguin_cache['items']:
            if item['itemId'].isdigit():
                item_map[item['itemId']] = item
        return item_map


item_map = load_item_map()


def get_all_item_img_in_screen(pil_screen):
    # 720p
    cv_screen = cv2.cvtColor(np.asarray(pil_screen), cv2.COLOR_BGR2RGB)
    dbg_screen = cv_screen.copy()
    img_h, img_w = cv_screen.shape[:2]
    ratio = 720 / img_h
    if ratio != 1:
        ratio = 720 / img_h
        cv_screen = cv2.resize(pil_screen, (int(img_w * ratio), 720))
    gray_screen = cv2.cvtColor(cv_screen, cv2.COLOR_BGR2GRAY)
    circles = get_circles(gray_screen)
    img_h, img_w = cv_screen.shape[:2]
    if circles is None:
        return []
    res = []
    for c in circles:
        x, y, box_size = int(c[0] - int(c[2] * 2.4) // 2), int(c[1] - int(c[2] * 2.4) // 2), int(c[2] * 2.4)
        if x > 0 and x + box_size < img_w:
            cv2.rectangle(dbg_screen, (x, y), (x + box_size, y + box_size), (7, 249, 151), 3)
            cv_item_img = cv_screen[y:y + box_size, x:x + box_size, :]
            cv_item_img2 = crop_item_middle_img(cv_item_img, c[2])
            num_img = crop_number_img(cv_item_img, c[2])
            # item_img = crop_item_img(cv_screen, gray_screen, c)
            res.append({'rectangle': cv_item_img, 'num_img': num_img, 'rectangle2': cv_item_img2})
    # show_img(dbg_screen)
    return res


def get_circles(gray_img, min_radius=55, max_radius=65):
    circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 100, param1=50,
                               param2=30, minRadius=min_radius, maxRadius=max_radius)
    return circles[0]


def crop_item_middle_img(cv_item_img, radius):
    img_h, img_w = cv_item_img.shape[:2]
    ox, oy = img_w // 2, img_h // 2
    ratio = radius / 60
    y1 = int(oy - (40 * ratio))
    y2 = int(oy + (24 * ratio))
    x1 = int(ox - (32 * ratio))
    x2 = int(ox + (32 * ratio))
    return cv2.resize(cv_item_img[y1:y2, x1:x2], (64, 64))


def crop_number_img(cv_item_img, radius):
    img_h, img_w = cv_item_img.shape[:2]
    ox, oy = img_w // 2, img_h // 2
    ratio = radius / 60
    y1 = int(oy + (30 * ratio))
    y2 = int(oy + (55 * ratio))
    x1 = int(ox - (20 * ratio))
    x2 = int(ox + (55 * ratio))
    return cv_item_img[y1:y2, x1:x2]


def hog(img):
    bin_n = 16
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  # hist is a 64 bit vector
    return hist


def get_img_feature(img):
    # return hog(img)
    return cv2.resize(img, (20, 20)).reshape((20*20, 1))


def cv_threshold(gray_img, threshold_val=127):
    res = cv2.threshold(gray_img, threshold_val, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if res[0][0] == 255:
        res = ~res
    return res


def crop_item_img(item_img, item_gray_img, circle):
    # circle = [x, y, radius]

    # show_img(item_img)
    # print(circle)
    mask = np.zeros_like(item_gray_img)
    cv2.circle(mask, (circle[0], circle[1]), int(circle[2]), 255, -1)
    out = np.zeros_like(item_img)
    # show_img(mask)
    out[mask == 255] = item_img[mask == 255]

    (y, x) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = out[topy:bottomy + 1, topx:bottomx + 1]
    img_h, img_w = out.shape[:2]
    out = out[0:img_h//2, 0:img_w]
    return out


def show_img(cv_img):
    cv2.imshow('test', cv_img)
    return cv2.waitKey(0)


def predict(circle_img):
    gray_img = cv2.cvtColor(circle_img, cv2.COLOR_BGR2GRAY)
    # show_img(gray_img)
    feature = get_img_feature(gray_img)
    res = svm.predict(np.float32([feature]))
    item_id = res[1][0][0]
    return item_map[item_id]


if __name__ == '__main__':
    from PIL import Image

    screen = Image.open('images/screen.png')
    item_images = get_all_item_img_in_screen(screen)
    show_img(item_images[0])
