import requests
from retry import retry


@retry(tries=10)
def request_get(url, print_resp=False, timeout=6):
    headers = {
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'DNT': '1',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Referer': 'http://prts.wiki/w/%E9%A6%96%E9%A1%B5',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }
    print('request_get:', url)
    if 'prts.wiki' in url:
        resp = requests.get(url, timeout=timeout, headers=headers)
    else:
        resp = requests.get(url, timeout=timeout)
    if print_resp:
        print(resp.content.decode('utf-8'))
    return resp
