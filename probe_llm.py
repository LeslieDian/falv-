# -*- coding: utf-8 -*-
"""端点自检：把本机已有的模型端点逐个试一遍，找出真正能用的那一个。

为什么需要它
------------
`config.json` 填错 base_url 或 key 时，页面上只会显示一句 `LLM HTTP 401/404`，
现场很难判断到底是哪一层错。这个脚本把「端点 + key + 模型名」三者一次性验完，
省得在答辩现场猜。

用法：
    .venv\\Scripts\\python.exe probe_llm.py            # 试 ~/.codebuddy* 里的全部端点
    .venv\\Scripts\\python.exe probe_llm.py --url ... --key ... --model ...
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

CONFIG_CANDIDATES = [
    os.path.expanduser("~/.codebuddycn/models.json"),
    os.path.expanduser("~/.codebuddy/models.json"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
]


def endpoint_of(url):
    url = (url or "").rstrip("/")
    return url if url.endswith("/chat/completions") else url + "/chat/completions"


def collect():
    items, seen = [], set()
    for p in CONFIG_CANDIDATES:
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            continue
        if p.endswith("config.json"):
            llm = d.get("llm") or {}
            if llm.get("api_key"):
                items.append((llm.get("model", "?"), llm["base_url"],
                              llm["api_key"], "config.json"))
            continue
        for m in d.get("models", []):
            url, key, mid = m.get("url") or "", m.get("apiKey") or "", m.get("id") or ""
            if not url or not key:
                continue
            sig = (url, mid)
            if sig in seen:
                continue
            seen.add(sig)
            items.append((mid, url, key, os.path.basename(os.path.dirname(p))))
    return items


def probe(model, url, key, timeout=45):
    body = {"model": model,
            "messages": [{"role": "user", "content": "用一句话说明什么是合同。"}],
            "max_tokens": 32, "stream": False}
    req = urllib.request.Request(
        endpoint_of(url), data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + key})
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
        j = json.loads(r.read().decode("utf-8", "replace"))
        ch = (j.get("choices") or [{}])[0]
        txt = (ch.get("message") or {}).get("content") or ch.get("text") or ""
        return "OK", txt.strip()[:40].replace("\n", " ")
    except urllib.error.HTTPError as e:
        return "HTTP %s" % e.code, e.read()[:110].decode("utf-8", "replace").replace("\n", " ")
    except Exception as e:                                         # noqa: BLE001
        return "ERR", str(e)[:90]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--key")
    ap.add_argument("--model")
    args = ap.parse_args()

    if args.url and args.key:
        items = [(args.model or "gpt-3.5-turbo", args.url, args.key, "命令行")]
    else:
        items = collect()

    if not items:
        print("没找到任何端点配置。请手动编辑 config.json 的 llm 段。")
        return 1

    print("=" * 78)
    print("  端点连通性自检（共 %d 个）" % len(items))
    print("=" * 78)
    good = []
    for mid, url, key, src in items:
        host = url.split("/")[2] if "//" in url else url
        status, msg = probe(mid, url, key)
        flag = "✓" if status == "OK" else " "
        print("%s %-24s %-34s %-8s %s" % (flag, mid, host, status, msg[:60]))
        if status == "OK":
            good.append((mid, url, key))

    print()
    if not good:
        print("没有可用端点。检查网络、key 是否过期、余额是否充足。")
        return 1
    print("可用的 %d 个：" % len(good))
    for mid, url, _ in good:
        print("  model = %-24s base_url = %s" % (mid, url))
    print("\n把上面任意一行的 base_url / model 填进 config.json 的 llm 段即可"
          "（key 用 .venv\\Scripts\\python.exe import_key.py 自动导入）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
