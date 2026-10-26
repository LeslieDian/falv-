# -*- coding: utf-8 -*-
"""本地知识库检索引擎（纯 numpy，无向量数据库依赖）。

设计取舍
--------
为什么不用 Chroma / FAISS：
  21.7 万条 × 1024 维的库，全量余弦只要一次分块矩阵乘（~0.2s），
  而且产物就是「一个 .npy + 两个 .jsonl」，拷进 U 盘就能走。
  引入 Chroma/FAISS 只会换来版本冲突、索引格式不兼容、迁移坑 —— 都是答辩现场
  最不该出现的东西。用户上传的增量单独存一个小 .npy，检索时与主库合并打分。

检索 = 向量召回 + 词面重排
--------------------------
纯向量的毛病：问「《刑法》第十七条」时它可能召回「第十八条」甚至别的法。
所以加两路轻量信号：
  1. 法名/条号精确命中  → 强 boost（法律问答里这是最可靠的锚点）
  2. 查询 2-gram 在文档中的覆盖率 → 弱重排
三者加权融合，不做花哨的 RRF。
"""
import json
import os
import re
import time
from functools import lru_cache

import numpy as np

CJK_KEEP = re.compile(r"[^\u4e00-\u9fff0-9a-zA-Z]+")
LAWNAME_RE = re.compile(r"《([^》]{2,40})》")
ARTNO_RE = re.compile(r"第([一二三四五六七八九十百零〇0-9]{1,6})条")

DOMAIN_LABEL = {
    "civil": "民事", "criminal": "刑事",
    "procedural": "程序", "general": "通用",
}

# 融合权重（向量 / 词面）
W_VEC = 0.70
W_LEX = 0.30
B_LAW = 0.25      # 法名命中
B_ART = 0.25      # 条号命中
B_LAW_ART = 0.20  # 法名 + 条号同时命中，额外再加
B_PREFER = 0.03   # 命中 L1 路由预测域的轻微加权（软偏向，不硬过滤）

# 案例任务类型的软惩罚 / 加权。
# 库内 14.5 万条案例来自 9 类指令数据集，其中**只有约四成**的参考答案是法律结论，
# 其余是标注型任务的产物（要素标注串、事件标签、匹配选项），拿来做「相似案例对比」
# 毫无价值 —— 甚至会让回答里出现「参考结论：标签:'婚后有子女'」这种内容。
# 按实测的可用性分档：
#   judgement_predit 判决推理（"根据《刑法》第133条…构成交通肇事罪"） → 最理想，加权
#   sent_pred        后果推断（法条 + 结论）                          → 可用，加权
#   op_sum / jud_doc_sum 摘要                                        → 可用，不罚
#   jud_read_compre  阅读理解（答案是日期/人名）                       → 中度惩罚
#   leg_case_cls     风险分类（答案是"灰产推广"）                      → 轻度惩罚
#   leg_ele_extra / leg_eve_detec / sim_case_match 标注型             → 重惩罚（近乎排除）
TASK_PENALTY = {
    "leg_ele_extra": 0.60,
    "leg_eve_detec": 0.60,
    "sim_case_match": 0.60,
    "jud_read_compre": 0.18,
    "leg_case_cls": 0.10,
    "jud_doc_sum": 0.05,
}
TASK_BONUS = {"judgement_predit": 0.03, "sent_pred": 0.02}

# 参考答案不是法律结论的类型：展示时不要把它当「参考结论」喂给模型
LABEL_TASKS = {"leg_ele_extra", "leg_eve_detec", "sim_case_match", "leg_case_cls"}
NO_CONCLUSION_HINT = "（该案例为标注型任务，原文未给出裁判结论，仅参考案件事实）"


def clean(s):
    return CJK_KEEP.sub("", s or "")


# ---------------------------------------------------------------- 模板清洗
# 案例语料来自 9 个公开指令数据集，24.5% 的记录里混着**数据集的任务模板**，
# 例如「请为下面的法律文书的案情描述片段贴上标签 …」「根据以下案例,回答下列问题。」
# 「你是一个法律顾问,现在有一些安卓隐私政策条款,请你提取出其中…」。
# 这些不是案情，会让模型以为用户给了个做题任务。
# 下面这批模板是**按全库开头片段的词频统计**建出来的（不是猜的），只做展示层清洗：
# 向量是按原文算的，不动数据、不重算向量。
_TPL_HEAD = [
    re.compile(r"^\s*请为下面的[^。\n]{0,60}?贴上标签"),
    re.compile(r"^\s*法律文书的案情描述片段[:：]?"),
    re.compile(r"^\s*你是一个法律顾问[^。\n]{0,240}?[。:：]"),
    re.compile(r"^\s*根据案例[ABC]的内容[^。\n]{0,90}?给出"),
    re.compile(r"^\s*请根据所给的法律案情内容[^。\n]{0,90}?[。:：]"),
    re.compile(r"^\s*请在下面的句子中识别出所有的法律要素[^。\n]{0,90}?[。:：]"),
    re.compile(r"^\s*请识别出以下法律案件中的所有事件触发词[^。\n]{0,90}?[。:：]"),
    re.compile(r"^\s*根据下面的法律案情段落内容判断案件中的被告人集合[^。\n]{0,40}?[。:：]?"),
    re.compile(r"^\s*请根据下面的案例描述对所提出的问题进行回答"),
    re.compile(r"^\s*以下文本反映了什么类型的风险"),
    re.compile(r"^\s*(?:阅读|根据)(?:下述|以下|所给)?(?:背景|案例|案件|材料)[^。\n]{0,14}?(?:回答|判断)[^。\n]{0,12}?[。:：?？]"),
    re.compile(r"^\s*根据案件背景[^。\n]{0,12}?[。:：?？]"),
    re.compile(r"^\s*以下是一个(?:法律案例|案例|例子)[:：]?"),
    re.compile(r"^\s*以下是一(?:起|篇|个)[^。\n]{0,16}?(?:案件|文书)[:：]?"),
    re.compile(r"^\s*这是一篇法律文书[:：]?"),
    re.compile(r"^\s*(?:基于|推理出|推测|预测|推断|评估)[^。\n]{0,30}?(?:判决|裁决|后果)[^。\n]{0,20}?[。:：?？]?"),
    re.compile(r"^\s*请(?:做出|预测)[^。\n]{0,20}?(?:判决|裁决)[^。\n]{0,10}?[。:：?？]?"),
    re.compile(r"^\s*在?法律三段论(?:中|是一种法理)[^。\n]{0,140}?[。:：]"),
    re.compile(r"^\s*请(?:大致描述|归纳)[^。\n]{0,24}?[。:：]"),
    re.compile(r"^\s*请完成[^。\n]{0,24}?任务[:：]?"),
    re.compile(r"^\s*请(?:从以下|根据以下|依据以下)[^。\n]{0,60}?[。:：]"),
    re.compile(r"^\s*回答(?:下列|下述)问题[。.]?"),
    re.compile(r"^\s*请回答[。.]?"),
    re.compile(r"^\s*贴上标签[。.]?"),
]
# 首行若命中这些特征，判定为「任务指令行」而不是案情
_INSTR_HINT = re.compile(
    r"(请|你是一个|根据|基于|依据|推理|推测|预测|推断|评估|阅读|以下|下述|这是|判断|识别|提取|描述|归纳|完成|回答|标注|风险|任务)")
_CASE_HEAD = re.compile(
    r"^(经审理|本院|公诉|原告|被告|上诉人|申请人|被害人|当事人|抗诉|自诉|再审|执行|19\d\d|20\d\d|一、|二、|1\.|\(20)")

_LEAD_LABEL = re.compile(r"^\s*(?:案例|案情|案件|背景|文本为|文本|原文|文段|材料)[:：]?\s*['\"“”]?")
_EDGE_QUOTE = re.compile(r"^[\s'\"“”‘’]+|[\s'\"“”‘’]+$")
_WS_RUN = re.compile(r"[ \t]{2,}")


def _is_instruction_line(line):
    """整行都是数据集的任务指令？（案情行一律以「经审理 / 本院 / 公诉机关 / 原告…」这类开头）"""
    s = line.strip()
    if not s or len(s) > 180:
        return False
    if _CASE_HEAD.match(s):
        return False
    if s.endswith(("?", "？", ":", "：")):
        return True
    return bool(_INSTR_HINT.search(s[:24]))


@lru_cache(maxsize=32768)
def strip_template(s):
    """去掉数据集任务模板，只留下真正的案情/结论正文。

    实测规律：模板与正文大多以**换行**分隔（如「推理出以下案件中的判决\\n禹城市人民检察院指控…」），
    少数用句号/冒号分隔。所以先按行丢弃指令行，再用锚定在开头的规则清理残留标签。

    保险：洗完若正文短于 60 字，说明可能误伤，退回原文（只做引号/标签清理）。
    """
    t = (s or "").strip()
    if not t:
        return t

    parts = t.split("\n")
    while len(parts) > 1 and _is_instruction_line(parts[0]) \
            and len("\n".join(parts[1:]).strip()) >= 60:
        parts.pop(0)
    out = "\n".join(parts)

    for p in _TPL_HEAD:
        out = p.sub(" ", out, count=1)

    out = _LEAD_LABEL.sub("", out.strip())
    out = _EDGE_QUOTE.sub("", out)
    out = _WS_RUN.sub(" ", out).strip()

    if len(out) < 60 <= len(t):
        out = _EDGE_QUOTE.sub("", _LEAD_LABEL.sub("", t).strip())
        out = _WS_RUN.sub(" ", out).strip()
    return out


def bigrams(s, limit=400):
    c = clean(s)
    if len(c) < 2:
        return {c} if c else set()
    if len(c) > limit:
        c = c[:limit]
    return set(c[i:i + 2] for i in range(len(c) - 1))


class KnowledgeBase(object):
    def __init__(self, kb_dir, log=None):
        self.kb_dir = os.path.abspath(kb_dir)
        self.log = log or (lambda *a: None)
        self.provisions = []
        self.cases = []
        self.vecs = None
        self.dim = None
        self.manifest = {}
        self.user_items = []          # 用户上传（含向量或仅词面）
        self.user_vecs = None
        self._loaded = False

    # ================================================================ 加载
    def load(self):
        t0 = time.time()
        mp = os.path.join(self.kb_dir, "manifest.json")
        if os.path.exists(mp):
            with open(mp, encoding="utf-8") as f:
                self.manifest = json.load(f)

        self.provisions = self._read_jsonl(os.path.join(self.kb_dir, "provisions.jsonl"))
        self.log("法条 %d 条" % len(self.provisions))
        self.cases = self._read_jsonl(os.path.join(self.kb_dir, "cases.jsonl"))
        self.log("案例 %d 条" % len(self.cases))

        vp = os.path.join(self.kb_dir, "vectors.f16.npy")
        if not os.path.exists(vp):
            raise SystemExit("缺少向量文件：%s" % vp)
        self.vecs = np.load(vp).astype(np.float16)
        self.dim = int(self.vecs.shape[1])
        expect = len(self.provisions) + len(self.cases)
        if self.vecs.shape[0] != expect:
            raise SystemExit("向量行数 %d 与文本条数 %d 不一致，知识库已损坏"
                             % (self.vecs.shape[0], expect))
        self.log("向量 %d × %d（%.0f MB），耗时 %.1fs"
                 % (self.vecs.shape[0], self.dim,
                    self.vecs.nbytes / 1048576, time.time() - t0))

        self._load_user()
        self._loaded = True
        return self

    @staticmethod
    def _read_jsonl(p):
        out = []
        if not os.path.exists(p):
            return out
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def _normalize_laws(self):
        """上游的 law_title 自带书名号（《中华人民共和国刑法》），
        而引用规范是「《法名》第X条」—— 这里统一去掉外层《》，
        否则渲染出来是《《…》》而且法名比对会漏。
        """
        for p in self.provisions:
            law = (p.get("law") or "").strip()
            while law.startswith("《") and law.endswith("》") and len(law) > 2:
                law = law[1:-1].strip()
            p["law"] = law
            if p.get("label"):
                p["label"] = p["label"].strip()

    # ------------------------------------------------------- 用户上传的增量
    @property
    def user_dir(self):
        d = os.path.join(self.kb_dir, "user")
        os.makedirs(d, exist_ok=True)
        return d

    def _load_user(self):
        self.user_items = self._read_jsonl(os.path.join(self.user_dir, "items.jsonl"))
        vp = os.path.join(self.user_dir, "vectors.f32.npy")
        if self.user_items and os.path.exists(vp):
            v = np.load(vp).astype(np.float32)
            if v.shape[0] == len(self.user_items):
                self.user_vecs = v
            else:
                self.user_vecs = None
        else:
            self.user_vecs = None
        if self.user_items:
            self.log("用户上传 %d 条（%s）"
                     % (len(self.user_items),
                        "含向量" if self.user_vecs is not None else "仅词面"))

    def add_user_item(self, item, vec=None):
        """上传一条。vec 为 None 时只进词面索引（降级模式仍可用）。"""
        self.user_items.append(item)
        if vec is not None:
            v = np.asarray(vec, dtype=np.float32).reshape(1, -1)
            self.user_vecs = v if self.user_vecs is None else \
                np.vstack([self.user_vecs, v])
        self._persist_user()
        return len(self.user_items)

    def _persist_user(self):
        p = os.path.join(self.user_dir, "items.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for it in self.user_items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        if self.user_vecs is not None:
            np.save(os.path.join(self.user_dir, "vectors.f32.npy"), self.user_vecs)

    def delete_user_item(self, idx):
        if not (0 <= idx < len(self.user_items)):
            return False
        self.user_items.pop(idx)
        if self.user_vecs is not None and self.user_vecs.shape[0] > idx:
            self.user_vecs = np.delete(self.user_vecs, idx, axis=0)
            if self.user_vecs.shape[0] == 0:
                self.user_vecs = None
        self._persist_user()
        return True

    # ================================================================ 统计
    def stats(self):
        prov_dom = {}
        law_cnt = {}
        for p in self.provisions:
            prov_dom[p.get("domain") or "?"] = prov_dom.get(p.get("domain") or "?", 0) + 1
            law = (p.get("law") or "").strip()
            if law:
                law_cnt[law] = law_cnt.get(law, 0) + 1
        case_dom = {}
        for c in self.cases:
            case_dom[c.get("domain") or "?"] = case_dom.get(c.get("domain") or "?", 0) + 1
        top_laws = sorted(law_cnt.items(), key=lambda kv: -kv[1])[:12]
        return {
            "n_provisions": len(self.provisions),
            "n_cases": len(self.cases),
            "n_vectors": 0 if self.vecs is None else int(self.vecs.shape[0]),
            "n_law_titles": len(law_cnt),
            "top_laws": [{"law": k, "n": v} for k, v in top_laws],
            "dim": self.dim,
            "n_user": len(self.user_items),
            "provision_domains": prov_dom,
            "case_domains": case_dom,
            "manifest": self.manifest,
        }

    # ================================================================ 打分
    def _vec_scores(self, qvec, block=65536):
        n = self.vecs.shape[0]
        out = np.empty(n, dtype=np.float32)
        for i in range(0, n, block):
            out[i:i + block] = self.vecs[i:i + block].astype(np.float32) @ qvec
        return out

    @staticmethod
    def _parse_query(query):
        return (bigrams(query), LAWNAME_RE.findall(query), ARTNO_RE.findall(query))

    @staticmethod
    def _lex(qbg, qlaws, qarts, doc_text, doc_law="", doc_label=""):
        c = clean(doc_text[:800])
        hit = 0
        for g in qbg:
            if g in c:
                hit += 1
        lex = float(hit) / max(1, len(qbg))
        boost = 0.0
        for L in qlaws:
            if not L or not doc_law:
                continue
            if L in doc_law or doc_law in L:
                boost += B_LAW
                break
        art_hit = False
        for a in qarts:
            if doc_label and a in doc_label:
                art_hit = True
                break
        if art_hit:
            boost += B_ART
            if boost >= B_LAW:
                boost += B_LAW_ART
        return lex, min(boost, 0.75)

    # ================================================================ 检索
    def retrieve(self, query, qvec=None, n_prov=8, n_case=6,
                 domains=None, pool=400, prefer=None):
        """→ dict(provisions=[...], cases=[...], user=[...], timing={...})

        prefer: L1 路由器预测出的域。命中该域的候选加一个小权重（软偏向）——
                不用硬过滤是因为路由本身只有 ~84% 准确，硬过滤会把错的那 16%
                变成"彻底检索不到"，代价太大。
        """
        t0 = time.time()
        qbg, qlaws, qarts = self._parse_query(query)
        dom = set(domains) if domains else None

        prov_scores = None
        case_scores = None
        if qvec is not None and self.vecs is not None:
            allsc = self._vec_scores(np.asarray(qvec, dtype=np.float32))
            n_p = len(self.provisions)
            prov_scores = allsc[:n_p]
            case_scores = allsc[n_p:]
        t_vec = time.time() - t0

        t1 = time.time()
        provs = self._rank(
            self.provisions, prov_scores, qbg, qlaws, qarts, n_prov, dom, pool,
            text_key="text", law_key="law", label_key="label", prefer=prefer)
        cases = self._rank(
            self.cases, case_scores, qbg, qlaws, qarts, n_case, dom,
            max(pool, 1500),          # 案例里七成是标注型任务，候选池开大一些，
                                      # 免得惩罚之后凑不满 n_case 条有用的判决类案例
            text_key="fact", law_key=None, label_key=None,
            fallback_limit=30000, prefer=prefer, task_key="task")
        user = self._rank_user(query, qvec, qbg, qlaws, qarts, dom)
        t_lex = time.time() - t1

        return {"provisions": provs, "cases": cases, "user": user,
                "timing": {"vector_ms": round(t_vec * 1000, 1),
                           "rank_ms": round(t_lex * 1000, 1),
                           "total_ms": round((time.time() - t0) * 1000, 1)}}

    def _rank(self, items, scores, qbg, qlaws, qarts, topk, dom, pool,
              text_key, law_key, label_key, fallback_limit=None, prefer=None,
              task_key=None):
        n = len(items)
        if n == 0:
            return []
        if scores is None:
            # 降级模式（编码器不可用）：全库词面扫描。
            # 法条文本短、库也小，全扫很便宜；案例库 14.5 万条全扫要十几秒，
            # 所以设一个上限并让前端提示"检索不完整"。
            lim = n if fallback_limit is None else min(n, fallback_limit)
            cand = range(lim)
        else:
            k = min(pool, n)
            idx = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
            cand = idx

        rows = []
        for i in cand:
            it = items[i]
            if dom and (it.get("domain") or "general") not in dom:
                continue
            lex, boost = self._lex(
                qbg, qlaws, qarts, it.get(text_key) or "",
                it.get(law_key) or "" if law_key else "",
                it.get(label_key) or "" if label_key else "")
            v = float(scores[i]) if scores is not None else 0.0
            if prefer and (it.get("domain") or "") == prefer:
                boost += B_PREFER
            if scores is not None:
                score = W_VEC * ((v + 1.0) / 2.0) + W_LEX * lex + boost
            else:
                score = lex + boost          # 纯词面模式
            if task_key:
                tk = it.get(task_key) or ""
                score -= TASK_PENALTY.get(tk, 0.0)
                score += TASK_BONUS.get(tk, 0.0)
            rows.append((score, v, lex, boost, i, it))
        rows.sort(key=lambda r: -r[0])

        # ---- 去重 ----------------------------------------------------------
        # 语料层面存在副本：同一部法规/司法解释被多次导入（正文只差空白），
        # 以及同一法条的不同年份版本（法名无版本号，条号相同、正文不同）。
        # 实测某题 top-8 里有 6 条只是 3 条 —— 名额被副本吃掉，模型还会看到
        # 「同一条号、两段不同正文」的自相矛盾材料。
        # 这里在**排序之后**按「同一法条 / 同一案情」折叠，并把名额让给后面
        # 真正不同的候选；不修改任何数据文件，向量矩阵也无需重建。
        out = []
        seen_keys = {}
        for score, v, lex, boost, i, it in rows:
            k = self._dedup_key(it, law_key, label_key, text_key)
            if k is not None and k in seen_keys:
                seen_keys[k]["_dups"] = seen_keys[k].get("_dups", 1) + 1
                continue
            row = dict(it)
            if text_key:
                row[text_key] = strip_template(row.get(text_key) or "")
            if row.get("answer") is not None:
                row["answer"] = strip_template(row.get("answer") or "")
            # 标注型任务的「答案」是标签串，不是裁判结论，不能当参考结论展示
            if (row.get("task") or "") in LABEL_TASKS:
                row["answer"] = NO_CONCLUSION_HINT
            row["_idx"] = int(i)
            row["_score"] = round(score, 4)
            row["_cos"] = round(v, 4)
            row["_lex"] = round(lex, 4)
            row["_boost"] = round(boost, 3)
            row["_dups"] = 1
            out.append(row)
            if k is not None:
                seen_keys[k] = row
            if len(out) >= topk:
                break
        return out

    @staticmethod
    def _dedup_key(it, law_key, label_key, text_key):
        """同一实体在库内可能有多份副本，返回折叠用的键；无法判定时返回 None（不去重）。

        法条：法名 + 条号（同一部法的同一个条号只该有一条现行文本）。
        案例：全文去空白后的字符串（避免只比前缀把不同案子误并）。
        """
        if law_key:
            law = clean(it.get(law_key) or "")
            if not law:
                return None
            return (law, clean(it.get(label_key) or "") if label_key else "")
        # 案例：先用 strip_template 去掉模板前缀 —— 同一个案子常被同时收进多个集合
        # （general / civil / criminal / procedural），它们只差前置的任务指令串，
        # 按原文比对是「不同记录」，按清洗后比对才是同一条。
        body = clean(strip_template(it.get(text_key) or ""))
        return body if len(body) >= 30 else None

    def _rank_user(self, query, qvec, qbg, qlaws, qarts, dom):
        if not self.user_items:
            return []
        use_vec = (qvec is not None and self.user_vecs is not None)
        uv = None
        if use_vec:
            uv = self.user_vecs @ np.asarray(qvec, dtype=np.float32)
        rows = []
        for i, it in enumerate(self.user_items):
            if dom and (it.get("domain") or "general") not in dom:
                continue
            body = (it.get("text") or "") + " " + (it.get("title") or "")
            lex, boost = self._lex(qbg, qlaws, qarts, body,
                                   it.get("law") or "", it.get("label") or "")
            v = float(uv[i]) if uv is not None else 0.0
            # 用户上传的内容给一点加权：他自己喂的东西应该看得见
            score = (W_VEC * ((v + 1.0) / 2.0) + W_LEX * lex + boost) * 1.03
            rows.append((score, v, lex, boost, i, it))
        rows.sort(key=lambda r: -r[0])
        out = []
        for score, v, lex, boost, i, it in rows[:4]:
            row = dict(it)
            row["_idx"] = -1 - int(i)     # 负数标记为「用户上传」
            row["_user_index"] = int(i)
            row["_score"] = round(score, 4)
            row["_cos"] = round(v, 4)
            row["_lex"] = round(lex, 4)
            row["_boost"] = round(boost, 3)
            out.append(row)
        return out
