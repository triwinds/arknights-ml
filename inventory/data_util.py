import json

with open('items.json', 'r', encoding='utf-8') as f:
    items = json.load(f)['data']


def get_ignore_item_ids():
    icon_map = {}
    ignore_item_ids = ['act13side_token_model_rep_1']
    for item in items:
        tmp = icon_map.get(item['iconId'], [])
        tmp.append(item['itemId'])
        icon_map[item['iconId']] = tmp
    for icon_id, item_ids in icon_map.items():
        # 罗德岛物资补给
        if len(item_ids) > 1 and icon_id.startswith('randomMaterial'):
            ignore_item_ids.extend(item_ids[:-1])
    return set(ignore_item_ids)


if __name__ == '__main__':
    print(get_ignore_item_ids())
