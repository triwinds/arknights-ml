import hashlib
import json
import os
import re
from functools import lru_cache

import bs4

from event_util import handle_special_item
from net_util import request_get

collect_path = 'images/collect/'


def update_items():
    global items
    print('update_items')
    resp = request_get(
        'https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData/master/zh_CN/gamedata/excel/item_table.json')
    md5 = hashlib.md5()
    md5.update(resp.content)
    items_map = resp.json()['items']
    items = [item for item in items_map.values()]
    data = {
        'hash': md5.hexdigest(),
        'data': items
    }
    remove_flag = False
    if os.path.exists('items.json'):
        with open('items.json', 'r', encoding='utf-8') as f:
            old_data = json.load(f)
            if old_data.get('hash') != data['hash']:
                remove_flag = True
    else:
        remove_flag = True

    if remove_flag:
        print('remove old collect')
        import clear_collect
        clear_collect.clear_collect_folder(collect_path)

    with open('items.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    items = data['data']
    return data['data']


def get_items():
    if not os.path.exists('items.json'):
        return update_items()
    else:
        with open('items.json', 'r', encoding='utf-8') as f:
            return json.load(f)['data']


items = get_items()


def get_items_id_map():
    res = {}
    for item in items:
        res[item['itemId']] = item
    return res


@lru_cache(1)
def get_items_name_map():
    res = {}
    for item in items:
        res[item['name']] = item
    return res


def download_icons():
    update_items()
    flag1 = download_from_items_page()
    flag2 = download_latest_event_icons()
    return flag1 or flag2


def download_from_items_page():
    print('checking item page...')
    resp = request_get('https://prts.wiki/w/%E9%81%93%E5%85%B7%E4%B8%80%E8%A7%88')
    soup = bs4.BeautifulSoup(resp.text, features='html.parser')
    data_divs = soup.find_all("div", {"class": "smwdata"})
    # print(data_devs[0])
    total = len(data_divs)
    c = 0
    update_flag = False
    for data_div in data_divs:
        if '分类:其他道具' in data_div['data-category']:
            print('skip', data_div['data-name'])
            continue
        item_name = data_div['data-name']
        flag = save_item(item_name, data_div['data-file'])
        if flag:
            update_flag = True
            print(item_name)
        c += 1
        # print(f'{c}/{total} {item_name}')
    return update_flag


def download_latest_event_icons():
    print('checking event page...')
    resp = request_get('https://prts.wiki/w/%E6%B4%BB%E5%8A%A8%E4%B8%80%E8%A7%88')
    soup = bs4.BeautifulSoup(resp.text, features='html.parser')
    event_tags = soup.find_all(text=' 进行中')
    event_tags += soup.find_all(text='未开始')
    update_flag = False
    if event_tags:
        for event_tag in event_tags:
            a_tag = event_tag.parent.parent.find_previous_sibling('a')
            event_url = 'https://prts.wiki' + a_tag['href']
            print('handle event:', a_tag.text)
            flag = download_from_event_page(event_url)
            if flag:
                update_flag = True
    return update_flag


def download_from_event_page(event_url):
    resp = request_get(event_url)
    soup = bs4.BeautifulSoup(resp.text, features='html.parser')
    item_imgs = soup.find_all('img', attrs={'alt': re.compile('道具')})
    item_set = set()
    update_flag = False
    for item_img in item_imgs:
        # print(item_img)
        if item_img['alt'] in item_set:
            continue
        if item_img.get('data-srcset') is None:
            continue
        if 'a' != item_img.parent.name:
            continue
        item_set.add(item_img['alt'])
        item_name = item_img.parent['title']
        img_url = 'https://prts.wiki' + item_img['data-srcset'].split(', ')[-1][:-3]
        flag = save_item(item_name, img_url)
        if flag:
            update_flag = True
    return update_flag


def save_item(item_name, img_url):
    if img_url.startswith('//'):
        img_url = 'https:' + img_url
    info = handle_special_item(item_name, img_url)
    if info:
        item_id, item_name, img_url = info[0], info[1], info[2]
        return save_img(item_id, item_name, img_url)

    items_name_map = get_items_name_map()
    item = items_name_map.get(item_name)
    item_id = 'other'
    if item:
        if item['itemType'] in {'MATERIAL', 'ARKPLANNER', 'ACTIVITY_ITEM', 'VOUCHER_MGACHA', 'AP_SUPPLY'} or item['itemId'].isdigit():
            item_id = item['itemId']
        if item['itemType'] not in {'ACTIVITY_ITEM', 'VOUCHER_MGACHA', 'AP_SUPPLY'} and not item_id.isdigit():
            item_id = 'other'
    return save_img(item_id, item_name, img_url)


def save_img(item_id, item_name, img_url):
    if img_url == '':
        print(f'skip {item_name}, img_url: {img_url}')
        return False
    if '家具收藏包' in item_name:
        print(f'skip 家具收藏包 [{item_name}], img_url: {img_url}')
        return False
    if not os.path.exists(collect_path + item_id):
        os.mkdir(collect_path + item_id)
    filepath = collect_path + item_id + '/%s.png' % item_name
    if os.path.exists(filepath):
        return False
    print(f'downloading {item_id}/{item_name} ...')
    print(f'img_url: {img_url}')
    rc = 0
    while rc <= 3:
        try:
            resp = request_get(img_url)
            with open(filepath, 'wb') as f:
                f.write(resp.content)
            return True
        except Exception as e:
            print(e)
            rc += 1
    raise RuntimeError(f'save_img reach max retry count, {item_id, item_name, img_url}')


if __name__ == '__main__':
    download_icons()
    # print(download_latest_event_icons())
    # download_from_event_page('https://prts.wiki/w/%E5%A4%9A%E7%B4%A2%E9%9B%B7%E6%96%AF%E5%81%87%E6%97%A5')
