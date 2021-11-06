from functools import lru_cache

from net_util import request_get


@lru_cache(1)
def get_charm_name_map():
    resp = request_get(
        'https://raw.githubusercontent.com/Kengxxiao/ArknightsGameData/master/zh_CN/gamedata/excel/charm_table.json')
    data = resp.json()
    res = {}
    for charm in data['charmList']:
        res[charm['name']] = charm
    return res


def get_charm_info(charm, img_url):
    item_id = f"@charm_r{charm['rarity']}_p{charm['price']}"
    item_name = charm['name']
    return item_id, item_name, img_url


def handle_special_item(item_name, img_url):
    charm_name_map = get_charm_name_map()
    if item_name in charm_name_map:
        return get_charm_info(charm_name_map.get(item_name), img_url)
