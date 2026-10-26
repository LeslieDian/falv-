# -*- coding: utf-8 -*-
"""环境自检：不依赖外接 API，验证知识库、向量、编码器、检索链路是否就绪。

用法：
    .venv\\Scripts\\python.exe selftest.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

FAIL = []


def step(msg):
    print("\n" + "-" * 68)
    print(msg)
    print("-" * 68)


def main():
    print("=" * 68)
    print("  法律领域智能体 · 环境自检")
    print("=" * 68)

    # ---------------------------------------------------------------- 文件
    step("[1] 知识库文件")
    need = ["provisions.jsonl", "cases.jsonl", "vectors.f16.npy", "manifest.json"]
    for n in need:
        p = os.path.join("kb", n)
        ok = os.path.exists(p)
        size = os.path.getsize(p) / 1048576 if ok else 0
        print("  %-20s %s  %8.1f MB" % (n, "OK " if ok else "缺失", size))
        if not ok:
            FAIL.append("缺少 kb/%s" % n)

    mdl = os.path.join("models", "Qwen3-Embedding-0.6B")
    wt = os.path.join(mdl, "model.safetensors")
    ok = os.path.exists(wt)
    print("  %-20s %s  %8.1f MB"
          % ("编码器权重", "OK " if ok else "缺失",
             os.path.getsize(wt) / 1048576 if ok else 0))
    if not ok:
        FAIL.append("缺少 models/Qwen3-Embedding-0.6B/model.safetensors")

    # ---------------------------------------------------------------- 加载
    step("[2] 加载知识库")
    import retrieval as R
    t0 = time.time()
    try:
        kb = R.KnowledgeBase("kb", log=lambda *a: None).load()
    except SystemExit as e:
        print("  加载失败：%s" % e)
        FAIL.append(str(e))
        return report()
    print("  %d 法条 / %d 案例 / 向量 %s，用时 %.1fs"
          % (len(kb.provisions), len(kb.cases), kb.vecs.shape, time.time() - t0))
    st = kb.stats()
    print("  法条域分布：%s" % st["provision_domains"])
    print("  案例域分布：%s" % st["case_domains"])

    # ---------------------------------------------------------------- 编码器
    step("[3] 查询编码器")
    from embedder import Embedder
    t0 = time.time()
    emb = Embedder(HERE)
    info = emb.info()
    if info["available"]:
        print("  就绪：%s / %s / %d 维，加载 %.1fs"
              % (info["model"], info["device"], info["dim"], time.time() - t0))
        t0 = time.time()
        v = emb.encode_queries(["租房押金被无故扣留怎么办"])
        print("  单条查询编码 %.2fs，向量模长 %.4f（应≈1.0）"
              % (time.time() - t0, float((v[0] ** 2).sum() ** 0.5)))
    else:
        print("  不可用：%s" % info["error"])
        print("  → 系统会自动降级为关键词检索（能跑，但语义召回会明显变差）")
        FAIL.append("编码器不可用（降级模式）")

    # ---------------------------------------------------------------- 检索
    step("[4] 检索链路抽样")
    from prompt import LawIndex, build_messages, verify_citations, norm_artno
    li = LawIndex(kb.provisions)
    print("  规范法名 %d 个 / 别名 %d 个" % (len(li.full), len(li.alias)))

    queries = [
        "快递运输途中商品损坏，风险由谁承担？",
        "股东要求解散公司的条件是什么？",
        "盗窃未遂怎么处理？",
    ]
    for q in queries:
        qv = emb.encode_queries([q])[0] if emb.available else None
        r = kb.retrieve(q, qvec=qv, n_prov=3, n_case=2)
        print("\n  问：%s" % q)
        print("     模式 %s，耗时 %s ms"
              % ("vector+lexical" if qv is not None else "lexical-only（降级）",
                 r["timing"]["total_ms"]))
        for p in r["provisions"]:
            print("     法条 《%s》%s  cos=%.3f 分=%.3f"
                  % (p.get("law"), p.get("label"), p["_cos"], p["_score"]))
        for c in r["cases"]:
            print("     案例 〔%s〕cos=%.3f 分=%.3f  %s"
                  % (c.get("domain"), c["_cos"], c["_score"],
                     (c.get("fact") or "")[:34].replace("\n", "")))

        # 顺带验证提示词与核验能跑通（用假回答）
        msgs = build_messages(q, r["provisions"], r["cases"], r["user"])
        mats = set((p.get("law") or "", norm_artno(p.get("label"))) for p in r["provisions"])
        fake = "根据《%s》%s 的规定，应当承担责任。另参见《不存在的法》第九十九条。" % (
            r["provisions"][0].get("law"), r["provisions"][0].get("label")) if r["provisions"] else ""
        ver = verify_citations(fake, mats, li)
        n_bad = len(ver["missing"]) + len(ver["law_unresolved"])
        print("     提示词 %d 字 / 核验：落实 %d，疑似编造 %d，法名未解析 %d（两者之和应≥1）"
              % (len(msgs[1]["content"]), len(ver["found"]),
                 len(ver["missing"]), len(ver["law_unresolved"])))
        # 《不存在的法》解析不出发名，会落到 law_unresolved 而不是 missing，
        # 两者都属于"核验能识别出这段引用有问题"，任一命中即算通过。
        if n_bad == 0:
            FAIL.append("核验未识别出编造条文（%s）" % q)

    # ---------------------------------------------------------------- 上传
    step("[5] 上传入库（含增量向量）")
    item = {"title": "自检：测试案例", "text": "甲将房屋出租给乙，租期届满乙拒不腾退，甲诉请返还房屋并支付占用费。",
            "domain": "civil", "kind": "case", "law": "", "label": ""}
    vec = emb.encode_docs([item["title"] + "\n" + item["text"]])[0] if emb.available else None
    n = kb.add_user_item(item, vec)
    print("  已入库，累计自建 %d 条（%s）" % (n, "含向量" if vec is not None else "仅词面"))
    qv = emb.encode_queries(["租期届满承租人拒不腾退房屋怎么办"])[0] if emb.available else None
    r = kb.retrieve("租期届满承租人拒不腾退房屋怎么办", qvec=qv, n_prov=1, n_case=1)
    hit = [u for u in r["user"] if "自检" in (u.get("title") or "")]
    print("  回查自建内容：%s" % ("命中 ✓" if hit else "未命中 ✗"))
    if not hit:
        FAIL.append("上传后未能检索到自建内容")
    kb.delete_user_item(n - 1)
    print("  已清理自检条目")

    return report()


def report():
    print("\n" + "=" * 68)
    if FAIL:
        print("  自检完成：%d 项需要注意" % len(FAIL))
        for f in FAIL:
            print("    ! %s" % f)
    else:
        print("  全部通过 —— 可以启动服务了： start.bat")
    print("=" * 68)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
