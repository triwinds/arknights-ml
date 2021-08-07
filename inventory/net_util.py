import requests
from retry import retry


@retry(tries=10)
def request_get(url, print_resp=False, timeout=6):
    print('request_get:', url)
    resp = requests.get(url, timeout=timeout)
    if print_resp:
        print(resp.content.decode('utf-8'))
    return resp
