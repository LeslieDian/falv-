# -*- coding: utf-8 -*-
"""在独立集上验证「本地 L1 路由器」是否与服务器端同量级。

为什么值得单独跑一次
--------------------
论文里报了 top1_acc 0.8370（服务器 GPU 上算的）。本地换成 CPU + 同样的权重文件，
如果数字对不上，说明**本地这套推理链路有问题**（前缀加错、阈值口径不同、标签映射漏了），
那后面用它做域偏向检索就是错的。所以先证明复现，再让它参与演示。

用法：
    .venv\\Scripts\\python.exe verify_router.py --sample 24      # 分层抽样，推荐
    .venv\\Scripts\\python.exe verify_router.py --limit 1000     # 全量（本机 CPU 约 1 小时）

⚠️ 本机 CPU 上编码约 0.03 s/token，测试集文本普遍 200–600 token，
   所以单条约 6–15 秒。先跑 --sample 24 拿到一个具代表性的小样本结论，
   全量复现请放到服务器上跑（那边有 GPU）。
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

# 真实域 → 路由标签（与服务器 label_map 同口径）
DOMAIN_TO_LABEL = {"civil": "civil", "criminal": "criminal",
                   "procedural": "procedure", "general": "unified"}


def build_eval_text(r):
    """与服务器 build_eval_text 逐句一致。"""
    ins = (r.get("instruction") or "").strip()
    inp = (r.get("input") or "").strip()
    if ins and inp:
        return ins + "\n" + inp
    return (r.get("ask") or ins or inp).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="router/test.jsonl")
    ap.add_argument("--limit", type=int, default=1000,
                    help="用前 N 条（测试集是按域分块排列的，前 N 条不具代表性，"
                         "小样本请改用 --sample）")
    ap.add_argument("--sample", type=int, default=0,
                    help=">0 时按域分层随机抽 N 条（推荐，小样本也具代表性）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print("缺少 %s" % args.data)
        return 1

    # 先把全量读进来（1000 行，很快），再做抽样决策
    pool = []          # [(text, label, 剪短的问句)]
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = build_eval_text(r)
            dom = r.get("domain") or (r.get("domains") or [None])[0]
            lab = DOMAIN_TO_LABEL.get(str(dom or "").strip().lower())
            if not t or not lab:
                continue
            pool.append((t, lab, (r.get("input") or "").strip()[:40]))

    if args.sample and args.sample > 0:
        import random
        buckets = {}
        for rec in pool:
            buckets.setdefault(rec[1], []).append(rec)
        for v in buckets.values():
            random.Random(args.seed).shuffle(v)
        picked, i = [], 0
        order = sorted(buckets)
        while len(picked) < min(args.sample, len(pool)):
            added = False
            for lab in order:
                if i < len(buckets[lab]) and len(picked) < args.sample:
                    picked.append(buckets[lab][i])
                    added = True
            if not added:
                break
            i += 1
        rows = picked
    else:
        rows = pool[:args.limit]

    texts = [r[0] for r in rows]
    truth = [r[1] for r in rows]
    qs = [r[2] for r in rows]

    print("样本 %d 条%s" % (len(texts),
                          "（分层抽样 seed=%d）" % args.seed if args.sample else ""))
    import collections
    print("真实标签分布：%s" % dict(collections.Counter(truth)))

    from embedder import Embedder
    emb = Embedder(HERE)
    if not emb.available:
        print("编码器不可用：%s" % emb.error)
        return 1

    # ★ 路由侧不加 instruction 前缀 → 用 encode_docs
    t0 = time.time()
    X = emb.encode_docs(texts)
    dt = time.time() - t0
    print("编码完成 shape=%s 用时 %.1fs（%.1f 条/s）"
          % (X.shape, dt, len(texts) / max(dt, 1e-9)))

    from domain_router import DomainRouter
    rt = DomainRouter(os.path.join(HERE, "router", "router_head.pkl"))
    if not rt.ok:
        print(rt.error)
        return 1
    res = rt.route_matrix(X)

    n = len(res)
    top1_hit = sum(1 for r, t in zip(res, truth) if r["raw_label"] == t)
    sys_hit = sum(1 for r, t in zip(res, truth) if r["label"] == t)
    fb = sum(1 for r in res if r["fallback"])
    cross = sum(1 for r in res if r["fallback_reason"] == "cross_domain")
    lowc = sum(1 for r in res if r["fallback_reason"] == "low_confidence")

    print("\n" + "=" * 66)
    print("  本地复现结果（阈值 %.2f，标签 %s）" % (rt.threshold, rt.labels))
    print("=" * 66)
    print("  top1_acc（argmax 命中）      %.4f   ← 论文口径，服务器报 0.8370"
          % (top1_hit / n))
    print("  system_acc（含阈值回退后）    %.4f" % (sys_hit / n))
    print("  fallback_rate                %.4f  （跨域 %.4f / 低置信 %.4f）"
          % (fb / n, cross / n, lowc / n))

    # 逐类
    print("\n  逐类（top-1 视角）：")
    for lab in rt.labels:
        idx = [i for i, t in enumerate(truth) if t == lab]
        if not idx:
            continue
        hit = sum(1 for i in idx if res[i]["raw_label"] == lab)
        print("    %-10s n=%4d  命中 %4d  召回 %.4f"
              % (lab, len(idx), hit, hit / len(idx)))

    # 抽样看几条
    print("\n  抽样：")
    for i in range(0, min(n, 8)):
        print("    [%s→%s] 置信 %.3f%s  %s"
              % (truth[i], res[i]["label"], res[i]["confidence"],
                 " (回退:%s)" % res[i]["fallback_reason"] if res[i]["fallback"] else "",
                 qs[i]))

    rep = {"n": n, "threshold": rt.threshold,
           "sampling": ("stratified sample=%d seed=%d" % (args.sample, args.seed))
                       if args.sample else ("first %d rows" % args.limit),
           "top1_acc": round(top1_hit / n, 4),
           "system_acc": round(sys_hit / n, 4),
           "fallback_rate": round(fb / n, 4),
           "cross_domain_rate": round(cross / n, 4),
           "low_conf_rate": round(lowc / n, 4),
           "ref_server_top1_acc": 0.8370,
           "encoded_sec": round(dt, 1)}
    with open(os.path.join(HERE, "router", "local_verify.json"), "w",
              encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("\n  已写入 router/local_verify.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
