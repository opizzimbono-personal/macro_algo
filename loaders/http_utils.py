import time
import requests

DEFAULT_TIMEOUT = 20
DEFAULT_SLEEP = 0.1


class HTTPError(Exception):
    pass


def get_json(url, params=None, headers=None, sleep=DEFAULT_SLEEP):
    resp = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
    if not resp.ok:
        raise HTTPError(f"GET {resp.url} -> {resp.status_code}: {resp.text[:200]}")
    if sleep:
        time.sleep(sleep)
    return resp.json()


def get_csv_text(url, params=None, headers=None, sleep=DEFAULT_SLEEP):
    resp = requests.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
    if not resp.ok:
        raise HTTPError(f"GET {resp.url} -> {resp.status_code}: {resp.text[:200]}")
    if sleep:
        time.sleep(sleep)
    return resp.text
