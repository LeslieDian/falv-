# -*- coding: utf-8 -*-
"""L1 请求级法域路由器（论文 MoE 第一级的本地复刻）。

论文里的位置
------------
MoE 两级路由的**第一级**：进来一个问题，先判断它属于哪个法域，
再决定用哪个领域专家、检索时偏向哪个子库。

★ 与服务器端逐位一致的三个要点（少一个数字就对不上）
----------------------------------------------------
1. **编码时不加 instruction 前缀**。检索侧用 `encode_queries()`（带
   "Instruct: ... retrieve ..."），那是"找像法条的文本"的指令；判域用
   `encode_docs()`（无前缀）才是中性表示。这一点训练脚本专门强调过。
2. **输入构造 = `instruction + "\\n" + input`，不含 system**。理由是真实系统里
   路由发生在检索之前，那一刻还没有法条可喂。
   → 训练集里 88% 的样本 instruction 为空，所以现场直接拿问题文本当输入，
     与训练分布天然一致。
3. **四标签 + 阈值回退**：多标签 sigmoid（每类一个二分类器），
   任一类 ≥ 0.6 才算命中；≥2 类命中算跨域，最高分不足 0.6 算低置信，
   两种都回退到 unified。

标签空间：['civil', 'criminal', 'unified', 'procedure']
论文口径（服务器端独立集 1,000 题）：top1_acc 0.8370，本文件应能复现同一量级。
"""
import os
import pickle

import numpy as np

LABELS = ["civil", "criminal", "unified", "procedure"]

# 路由标签 → 知识库域（kb 里的域是 general/civil/criminal/procedural）
LABEL_TO_DOMAIN = {"civil": "civil", "criminal": "criminal",
                   "procedure": "procedural", "unified": "general"}
LABEL_CN = {"civil": "民事", "criminal": "刑事",
            "procedure": "程序", "unified": "通用"}
DOMAIN_CN = {"civil": "民事", "criminal": "刑事",
             "procedural": "程序", "general": "通用"}


class DomainRouter(object):
    def __init__(self, path, threshold=None):
        self.path = path
        self.ok = False
        self.error = None
        self.heads = []
        self.labels = LABELS
        self.threshold = 0.6
        self.label_map = {}
        self.load(threshold)

    def load(self, threshold=None):
        try:
            with open(self.path, "rb") as f:
                d = pickle.load(f)
            self.heads = d["heads"]
            self.labels = d["labels"]
            self.threshold = float(threshold if threshold is not None
                                   else d.get("threshold", 0.6))
            self.label_map = d.get("label_map") or {}
            self.ok = True
        except Exception as e:                                     # noqa: BLE001
            self.error = "路由器加载失败：%s" % e

    def info(self):
        return {"available": self.ok, "threshold": self.threshold,
                "labels": self.labels, "error": self.error}

    # ------------------------------------------------------------------ 推理
    def predict_proba(self, X):
        cols = []
        for kind, obj in self.heads:
            if kind == "const":
                cols.append(np.full(X.shape[0], float(obj), dtype="float64"))
            else:
                cols.append(obj.predict_proba(X)[:, 1])
        return np.vstack(cols).T

    def route_matrix(self, X):
        """→ list[dict]，与服务器 evaluate() 的决策口径一致。"""
        P = self.predict_proba(X)
        top1 = P.argmax(axis=1)
        n_over = (P >= self.threshold).sum(axis=1)
        cross = n_over >= 2
        low = P.max(axis=1) < self.threshold
        out = []
        for i in range(P.shape[0]):
            raw = self.labels[int(top1[i])]
            fb = bool(cross[i] or low[i])
            final = "unified" if fb else raw
            out.append({
                "label": final,
                "raw_label": raw,
                "domain": LABEL_TO_DOMAIN.get(final, "general"),
                "domain_cn": DOMAIN_CN.get(LABEL_TO_DOMAIN.get(final, "general"), "通用"),
                "confidence": round(float(P[i].max()), 4),
                "fallback": fb,
                "fallback_reason": ("cross_domain" if cross[i]
                                    else ("low_confidence" if low[i] else "")),
                "probs": {LABEL_CN[self.labels[j]]: round(float(P[i][j]), 4)
                          for j in range(len(self.labels))},
            })
        return out

    def route_one(self, x):
        return self.route_matrix(np.asarray(x, dtype="float64").reshape(1, -1))[0]

    def domain_of(self, x):
        return self.route_one(x)["domain"]


def load_router(base_dir, filename="router/router_head.pkl", threshold=None):
    p = os.path.join(base_dir, filename)
    if not os.path.exists(p):
        r = DomainRouter.__new__(DomainRouter)
        r.path = p; r.ok = False; r.error = "路由器文件不存在：%s" % p
        r.heads = []; r.labels = LABELS; r.threshold = 0.6; r.label_map = {}
        return r
    return DomainRouter(p, threshold)
