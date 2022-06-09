import hashlib
import json
import os
from functools import lru_cache

from github import Github

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


def get_icon_id_map():
    res = {}
    for item in items:
        tmp_list = res.get(item['iconId'], [])
        tmp_list.append(item)
        res[item['iconId']] = tmp_list
    return res


def download_icons():
    update_items()
    update_flag = False
    gh_item_files = list_item_dir()
    icon_id_map = get_icon_id_map()
    for content_file in gh_item_files:
        icon_id = content_file.name[:-4]
        tmp_list = icon_id_map.get(icon_id, [])
        if icon_id == 'randomMaterial_1':
            tmp_list = [tmp_list[-1]]
        for item_info in tmp_list:
            # print(icon_id, item_info)
            if save_item(item_info['name'], content_file.download_url):
                update_flag = True
    return update_flag


def list_item_dir():
    g = Github()
    repo = g.get_repo('yuanyan3060/Arknights-Bot-Resource')
    return repo.get_contents('item')


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
