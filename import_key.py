# -*- coding: utf-8 -*-
"""把本机已有的模型端点导入 config.json，省得手抄 key。

数据来源：CodeBuddy 的模型配置（~/.codebuddycn/models.json 或 ~/.codebuddy/models.json）。
只读取，不改动原文件；写入的 key 只落在本项目的 config.json。

用法：
    .venv\\Scripts\\python.exe import_key.py            # 列出候选，交互选择
    .venv\\Scripts\\python.exe import_key.py --index 0  # 直接选第 0 个
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [os.path.expanduser("~/.codebuddycn/models.json"),
              os.path.expanduser("~/.codebuddy/models.json")]


def mask(k):
    if not k:
        return "(空)"
    return k[:6] + "…" + k[-4:] if len(k) > 12 else "…"


def collect():
    out = []
    for p in CANDIDATES:
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception as e:                                     # noqa: BLE001
            print("[skip] %s 解析失败：%s" % (p, e))
            continue
        models = d.get("models") if isinstance(d, dict) else d
        if isinstance(models, dict):
            models = list(models.values())
        for m in models or []:
            if not isinstance(m, dict):
                continue
            url = m.get("url") or m.get("baseUrl") or m.get("base_url") or ""
            key = m.get("apiKey") or m.get("api_key") or ""
            name = m.get("name") or m.get("id") or ""
            mid = m.get("id") or m.get("name") or ""
            if url and key:
                out.append({"name": name, "url": url, "key": key,
                            "model": mid, "src": os.path.basename(os.path.dirname(p))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--name", default=None, help="按名称关键字选，如 --name MiniMax")
    args = ap.parse_args()

    items = collect()
    if not items:
        print("没找到可用的端点配置。请手动编辑 config.json 的 llm 段。")
        return 1

    print("=" * 70)
    print("  本机可用的模型端点")
    print("=" * 70)
    for i, it in enumerate(items):
        print("  [%2d] %-22s %s" % (i, it["name"], it["url"]))
        print("       key=%s  来源=%s" % (mask(it["key"]), it["src"]))
    print()

    idx = args.index
    if idx is None and args.name:
        hits = [i for i, it in enumerate(items)
                if args.name.lower() in (it.get("name") or "").lower()]
        if len(hits) == 1:
            idx = hits[0]
        elif not hits:
            print("没有名称包含 %r 的端点。" % args.name)
            return 1
        else:
            print("名称匹配到多个，请改用序号：")
            for i in hits:
                print("  [%d] %s" % (i, items[i]["name"]))
            return 1
    if idx is None:
        raw = input("输入序号（回车取消）：").strip()
        if not raw:
            print("已取消。")
            return 0
        idx = int(raw)
    if not (0 <= idx < len(items)):
        print("序号超出范围。")
        return 1

    it = items[idx]
    cfg_path = os.path.join(HERE, "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path, encoding="utf-8"))
    cfg.setdefault("llm", {})
    cfg["llm"]["base_url"] = it["url"]
    cfg["llm"]["api_key"] = it["key"]
    cfg["llm"]["model"] = it["model"]
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("\n已写入 config.json：")
    print("  base_url = %s" % it["url"])
    print("  model    = %s" % it["model"])
    print("  api_key  = %s" % mask(it["key"]))
    print("\n接着跑： .venv\\Scripts\\python.exe selftest.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
