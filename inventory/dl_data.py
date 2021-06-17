import requests
import bs4
import os
import json
import hashlib
import shutil
from retry import retry


collect_path = 'images/collect/'


@retry(tries=3)
def request_get(url):
    return requests.get(url)


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
        shutil.rmtree(collect_path)
        os.mkdir(collect_path)

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


def get_items_name_map():
    res = {}
    for item in items:
        res[item['name']] = item
    return res


def download_icons():
    update_items()
    items_name_map = get_items_name_map()
    resp = request_get('http://prts.wiki/w/%E9%81%93%E5%85%B7%E4%B8%80%E8%A7%88')
    soup = bs4.BeautifulSoup(resp.text, features='html.parser')
    data_devs = soup.find_all("div", {"class": "smwdata"})
    # print(data_devs[0])
    total = len(data_devs)
    c = 0
    update_flag = False
    for data_dev in data_devs:
        item_name = data_dev['data-name']
        item = items_name_map.get(item_name)
        item_id = 'other'
        if item and item['itemType'] in ['MATERIAL', 'ARKPLANNER', 'ACTIVITY_ITEM']:
            item_id = item['itemId']
            if item['itemType'] != 'ACTIVITY_ITEM' and not item_id.isdigit() or len(item_id) < 5:
                item_id = 'other'
        flag = save_img(item_id, item_name, data_dev['data-file'])
        if flag:
            update_flag = True
            print(item)
        c += 1
        # print(f'{c}/{total} {item_name}')
    return update_flag


def save_img(item_id, item_name, img_url):
    if img_url == '':
        print(f'skip {item_name}, img_url: {img_url}')
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
