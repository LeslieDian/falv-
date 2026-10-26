# -*- coding: utf-8 -*-
"""法律领域智能体 · 离线演示服务（本地跑，生成走外接 API）

形态
----
  本机一条命令起服务 → 浏览器打开 → 现场提任何问题（不限于题库）。
  向量库、法条、案例全部来自本地 kb/ 目录（从服务器导出，可随 U 盘迁移）。

三个必须说清楚的设计点
----------------------
1. **embedding 与生成是两件事**。检索用的向量必须由 Qwen3-Embedding-0.6B
   （随包附带）产生；生成可以换成任意 OpenAI 兼容 API。二者不能互相替代：
   用第三方 embedding API 去查 Qwen3 建的库，等于查错空间。
2. **生成与核查分离**。外接模型更强也更敢编，所以每条回答都要过一遍
   verify（prompt.py），把《法名》第X条逐条对回本次检索到的材料。
3. **没有 API key 也能演示**。此时只跑检索，把命中的法条与相似案例摊开给
   观众看 —— 检索层本来就是可验证的，生成层只是锦上添花。

启动
----
    python -m uvicorn app:app --host 127.0.0.1 --port 8760
  或直接  python run.py
"""
import json
import os
import random
import re
import time
import traceback
from typing import List, Optional

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import prompt as P
import retrieval as R
from embedder import Embedder
from domain_router import load_router

HERE = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(HERE, "kb")
WEB_DIR = os.path.join(HERE, "web")
CONFIG_PATH = os.path.join(HERE, "config.json")

DEFAULT_CONFIG = {
    "llm": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "",
        "model": "deepseek-chat",
        "temperature": 0.3,
        "max_tokens": 4096,
        "timeout": 120,
        "reasoning_split": True,
    },
    "retrieval": {"n_provisions": 8, "n_cases": 6, "pool": 400},
    "server": {"host": "127.0.0.1", "port": 8760},
    "embedding": {"model_dir": "", "device": "cpu", "batch_size": 16},
}


def load_config():
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                user = json.load(f)
            for k, v in user.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        except Exception as e:                                     # noqa: BLE001
            print("[warn] config.json 解析失败，用默认配置：%s" % e)
    # 环境变量优先（免改文件，方便临时换 key）
    for env, path in (("LAW_RAG_API_KEY", ("llm", "api_key")),
                      ("LAW_RAG_BASE_URL", ("llm", "base_url")),
                      ("LAW_RAG_MODEL", ("llm", "model"))):
        v = os.environ.get(env)
        if v:
            cfg[path[0]][path[1]] = v
    return cfg


CFG = load_config()
LOG = []

# 随机抽题时的案情清洗：只保留真实叙事，剔掉数据集指令
_NARRATIVE_RE = re.compile(
    r"(经审理查明|经审理后查明|本院查明|法院查明|人民检察院指控|公诉机关指控|"
    r"检察院指控|原告.{0,8}诉称|自诉人.{0,8}诉称|上诉人.{0,8}诉称|案情[:：]|事实[:：])")
_INSTRUCTION_RE = re.compile(
    r"(请[为从根据分析对][^。\n]{0,45}|回答下列问题[^。\n]{0,20}|贴上标签[^。\n]{0,20}|"
    r"推理出[^。\n]{0,20}|法律文书的案情描述片段)")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.append("%s %s" % (time.strftime("%H:%M:%S"), msg))
    del LOG[:-200]


# ======================================================================= 状态
class State(object):
    kb = None
    emb = None
    law_index = None
    error = None


def boot():
    t0 = time.time()
    try:
        log("[boot] 加载知识库 %s" % KB_DIR)
        State.kb = R.KnowledgeBase(KB_DIR, log=log).load()
        State.law_index = P.LawIndex(State.kb.provisions)
        log("[boot] 法名索引 %d 个规范法名 / %d 条别名"
            % (len(State.law_index.full), len(State.law_index.alias)))
        md = CFG.get("embedding", {}).get("model_dir") or None
        State.emb = Embedder(HERE, model_dir=md,
                             device=CFG.get("embedding", {}).get("device", "cpu"),
                             batch_size=CFG.get("embedding", {}).get("batch_size", 16))
        info = State.emb.info()
        if info["available"]:
            log("[boot] 查询编码器就绪：%s / %s / %dd"
                % (info["model"], info["device"], info["dim"]))
        else:
            log("[boot] !! 查询编码器不可用 → 降级为关键词检索：%s" % info["error"])
        State.router = load_router(HERE)
        ri = State.router.info()
        if ri["available"]:
            log("[boot] L1 法域路由器就绪：标签 %s / 阈值 %.2f"
                % (ri["labels"], ri["threshold"]))
        else:
            log("[boot] L1 路由器不可用：%s" % ri["error"])
        log("[boot] 就绪，用时 %.1fs" % (time.time() - t0))
    except Exception as e:                                         # noqa: BLE001
        State.error = "%s\n%s" % (e, traceback.format_exc())
        log("[boot] 失败：%s" % e)


boot()

app = FastAPI(title="法律领域智能体 · 离线演示")


def _llm_ready():
    return bool(CFG["llm"].get("api_key"))


def _client(timeout=None):
    # connect 放宽到 30s：推理模型首字节慢，且答辩现场网络一般，
    # connect=15 容易在网络抖动时直接 ConnectTimeout，链路整条掉下去。
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=30.0, read=float(timeout or CFG["llm"]["timeout"]),
                              write=60.0, pool=30.0),
        follow_redirects=True)


def _endpoint():
    """容忍用户把 base_url 填成完整地址（很多中转商给的就是 .../v1/chat/completions）。"""
    base = CFG["llm"]["base_url"].rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _headers():
    h = {"Content-Type": "application/json"}
    if CFG["llm"].get("api_key"):
        h["Authorization"] = "Bearer " + CFG["llm"]["api_key"]
    return h


# 推理模型（如 MiniMax-M3、DeepSeek-R1）把思维链内联在 content 里，
# 且**思维链和正文共用 max_tokens 预算**。预算给小了会出现一个很难查的故障：
# 模型把 token 全花在 <think>…</think> 里，正文一个字都没输出，
# 页面上只看到"模型未返回正文"—— 而接口其实返回了 HTTP 200。
# 下面两个措施一起上：
#   1) reasoning_split=True —— 让服务端把思维链放到独立字段，不与正文抢预算；
#   2) max_tokens 取配置值与下限的较大者 —— 给思维链留够空间。
# 不支持该字段的服务商会忽略它（OpenAI 兼容接口普遍如此），不影响。
_REASONING_FLOOR = 8192


def _llm_body(msgs, stream=False):
    """统一构造发给 OpenAI 兼容接口的请求体。"""
    body = {"model": CFG["llm"]["model"],
            "messages": msgs,
            "temperature": CFG["llm"]["temperature"],
            "max_tokens": max(int(CFG["llm"]["max_tokens"]), _REASONING_FLOOR),
            "stream": stream}
    if CFG["llm"].get("reasoning_split", True):
        body["reasoning_split"] = True
    return body


# ------------------------------------------------------- 兜底回答（不依赖外网）
def local_answer(question, ret, reason=""):
    """外接模型不可用时，用**本地检索材料**拼一份结构化证据摘要。

    为什么要有这个
    --------------
    答辩现场三种情况都会让生成层掉线：没网 / 中转服务失联 / 额度用尽。
    此时如果只弹一条红色报错，观众看到的就是「系统坏了」。而检索层本来
    不依赖外网（纯 numpy 读本地向量），所以这里直接把命中的法条与相似
    案例摊开成一份可读的回答 —— 链路仍然是活的，只是少了大模型的推理。

    这份内容**不做任何推断**，只复述库内原文，因此不存在编造风险。
    """
    provs = ret.get("provisions") or []
    cases = ret.get("cases") or []
    L = []
    L.append("> ⚠ **当前为本地检索模式**：外接生成模型不可用%s。"
             "下面给出的法条与案例均直接取自本地知识库原文（未做推断），"
             "修好模型配置后同一问题会自动补上要件分析与结论。"
             % (("（%s）" % reason) if reason else ""))
    L.append("")
    L.append("## 一、问题")
    L.append(question.strip())
    L.append("")
    L.append("## 二、检索到的相关法条（共 %d 条，按相关度排序）" % len(provs))
    for i, p in enumerate(provs[:6], 1):
        law = (p.get("law") or "").strip()
        label = (p.get("label") or "").strip()
        body = re.sub(r"\s+", " ", (p.get("text") or "").strip())
        L.append("**%d. 《%s》%s**（相似度 %.3f）" % (i, law, label, float(p.get("_cos") or 0)))
        L.append(body[:260] + ("…" if len(body) > 260 else ""))
        L.append("")
    if cases:
        L.append("## 三、库内相似案例（共 %d 条）" % len(cases))
        for i, c in enumerate(cases[:3], 1):
            fact = re.sub(r"\s+", " ", (c.get("fact") or "").strip())
            L.append("**%d.** 〔%s〕相似度 %.3f —— %s"
                     % (i, c.get("domain") or "?", float(c.get("_cos") or 0),
                        fact[:200] + ("…" if len(fact) > 200 else "")))
            L.append("")
    L.append("## 四、说明")
    L.append("本次回答由**检索层独立产出**：全库 %d 条向量一次性打分，"
             "未调用任何外部服务，断网亦可复现。" % (ret.get("n_total") or 0))
    return "\n".join(L)


# ======================================================================= 检索
def do_retrieve(question, domains=None, n_prov=None, n_case=None, retr_query=None):
    """retr_query：多轮场景下拼接了上一轮用户问题的生效查询（缺省用 question 本身）。"""
    kb = State.kb
    q_retr = retr_query or question
    qvec = None
    t0 = time.time()
    if State.emb is not None and State.emb.available:
        qvec = State.emb.encode_queries([q_retr])[0]
    t_emb = time.time() - t0

    # ---- L1 请求级法域路由（论文 MoE 第一级）-------------------------------
    # 路由器输入用 encode_docs（**不加** retrieval instruction），
    # 与训练脚本一致；它和上面的检索 query 是两个不同的表示，必须分开编码。
    route = None
    if (State.router is not None and State.router.ok
            and State.emb is not None and State.emb.available):
        try:
            t1 = time.time()
            xr = State.emb.encode_docs([q_retr])[0]
            route = State.router.route_one(xr)
            route["ms"] = round((time.time() - t1) * 1000, 1)
        except Exception as e:                                     # noqa: BLE001
            route = {"error": "%s: %s" % (type(e).__name__, e)}

    # 回退（跨域/低置信）时不施加域偏向 —— 回退本身意味着"拿不准"，
    # 再偏向 general 库只是白白引入偏差。
    prefer = None
    if route and route.get("domain") and not route.get("fallback"):
        prefer = route["domain"]

    rc = CFG["retrieval"]
    res = kb.retrieve(question, qvec=qvec,
                      n_prov=n_prov or rc["n_provisions"],
                      n_case=n_case or rc["n_cases"],
                      domains=domains, pool=rc.get("pool", 400), prefer=prefer)
    res["timing"]["embed_ms"] = round(t_emb * 1000, 1)
    res["mode"] = "vector+lexical" if qvec is not None else "lexical-only"
    res["route"] = route
    res["n_total"] = len(kb.provisions) + len(kb.cases)
    return res


# ------------------------------------------------------- 回答附带的 Top-K 案例
def _top_cases(ret, k):
    """挑出回答下方要展示的 Top-K 相似案例。

    为什么单独挑一遍而不是直接用 ret["cases"]
    ----------------------------------------
    ret["cases"] 是**喂给模型**的材料，条数由界面上的「案例」参数决定（默认 6），
    里面混着「标注型任务」的条目（它们的 answer 字段是标签串，不是裁判结论）。
    而回答下方的 Top-K 是给用户看的「参考判例」，标准更严：
      1. 优先要**有裁判结论**的案例（answer 非空、且不是标注占位符）；
      2. 再按相关度排序取前 k 条；
      3. k=0 时不展示。
    这样「回答里的案例引用」与「界面上的参考判例」两件事不会互相干扰。
    """
    k = int(k or 0)
    if k <= 0:
        return []
    pool = list(ret.get("cases") or [])
    # 有结论的优先，但保持相关度顺序（ret["cases"] 本身就是按分数排好的）
    with_ans = [c for c in pool
                if (c.get("answer") or "").strip()
                and not str(c.get("answer")).strip().startswith("（无")]
    chosen = with_ans[:k] if len(with_ans) >= 1 else pool[:k]
    out = []
    for i, c in enumerate(chosen, 1):
        fact = re.sub(r"\s+", " ", (c.get("fact") or "").strip())
        out.append({
            "rank": i,
            "title": (c.get("title") or "").strip() or "〔%s〕案例" % (c.get("domain") or "库内"),
            "domain": c.get("domain") or "",
            "task": c.get("task") or "",
            "fact": fact[:400],
            "answer": (c.get("answer") or "").strip()[:300],
            "cos": c.get("_cos"),
            "score": c.get("_score"),
        })
    return out


# ======================================================================= 路由
@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/api/stats")
def api_stats():
    if State.error or State.kb is None:
        return JSONResponse({"ok": False, "error": State.error or "知识库未加载"}, 500)
    s = State.kb.stats()
    s["ok"] = True
    s["embedder"] = State.emb.info() if State.emb else None
    s["router"] = State.router.info() if State.router else None
    s["llm"] = {"ready": _llm_ready(), "model": CFG["llm"]["model"],
                "base_url": CFG["llm"]["base_url"]}
    s["n_laws"] = len(State.law_index.full)
    s["log"] = LOG[-12:]
    return s


@app.get("/api/random_case")
def api_random_case(domain: str = ""):
    """从库内**真实案例**里随机抽一条。

    用途：现场演示时点一下就能抽到一个没预设过的案子 —— 直接证明
    「不是回放预制答案，全库 14.5 万案例随便抽都能推理」。

    注意：库内案例的 fact 字段有些自带数据集指令（「请为下面的法律文书…贴上标签」），
    直接投喂很难看。这里只挑带真实叙事标记的，并从标记处开始截取。
    """
    if State.kb is None or not State.kb.cases:
        return JSONResponse({"ok": False, "error": "知识库未加载"}, 500)
    pool = State.kb.cases
    if domain:
        sub = [c for c in pool if (c.get("domain") or "") == domain]
        pool = sub or pool
    # 优先挑有真实案情的（含「检察院指控 / 经审理查明 / 诉称」这类标记）
    rich = [c for c in pool if _NARRATIVE_RE.search(c.get("fact") or "")]
    picked = random.choice(rich or pool)
    fact = (picked.get("fact") or "").strip()
    start = _NARRATIVE_RE.search(fact)
    if start:
        fact = fact[start.start():]
    fact = _INSTRUCTION_RE.sub("", fact).strip()
    return {"ok": True, "domain": picked.get("domain") or "",
            "task": picked.get("task") or "",
            "fact": fact[:320], "n_pool": len(pool), "n_rich": len(rich),
            "has_answer": bool((picked.get("answer") or "").strip())}


class AskReq(BaseModel):
    question: str
    domains: Optional[List[str]] = None
    n_provisions: Optional[int] = None
    n_cases: Optional[int] = None
    history: Optional[List[dict]] = None
    detailed: bool = False        # True → 结构化详细分析（JSON）
    top_k_cases: int = 3          # 回答下方附带的 Top-K 相似案例条数


@app.post("/api/retrieve")
async def api_retrieve(req: AskReq):
    """只跑检索、不调生成 —— 快（2~3s），用于覆盖面核查与演示「检索层可验证」。"""
    if State.kb is None:
        return JSONResponse({"ok": False, "error": State.error or "知识库未加载"}, 500)
    hist = P.clip_history(req.history)
    q_retr = P.effective_query(req.question, hist)
    ret = await run_in_threadpool(do_retrieve, req.question, req.domains,
                                  req.n_provisions, req.n_cases, q_retr)
    return {"ok": True, "evidence": ret}


@app.post("/api/ask")
async def api_ask(req: AskReq):
    if State.kb is None:
        return JSONResponse({"ok": False, "error": State.error or "知识库未加载"}, 500)
    t0 = time.time()
    hist = P.clip_history(req.history)
    q_retr = P.effective_query(req.question, hist)
    ret = await run_in_threadpool(do_retrieve, req.question, req.domains,
                                  req.n_provisions, req.n_cases, q_retr)
    ret["top_cases"] = _top_cases(ret, req.top_k_cases)
    answer = ""
    err = None
    if _llm_ready():
        msgs = P.build_messages(req.question, ret["provisions"],
                                ret["cases"], ret["user"], history=hist,
                                detailed=req.detailed)
        try:
            async with _client() as c:
                r = await c.post(_endpoint(), headers=_headers(),
                                 json=_llm_body(msgs, stream=False))
                if r.status_code != 200:
                    err = "LLM HTTP %d: %s" % (r.status_code, r.text[:400])
                else:
                    j = r.json()
                    msg = (j.get("choices") or [{}])[0].get("message") or {}
                    content = msg.get("content") or ""
                    # reasoning_split 开启时思维链在 reasoning_content，正文干净；
                    # 未开启时思维链内联在 content 里，需要 strip_think 剥掉。
                    answer = strip_think(content)
                    if not answer.strip():
                        answer = (msg.get("reasoning_content") or "") and strip_think(
                            msg.get("reasoning_content") or "") or answer
        except Exception as e:                                     # noqa: BLE001
            err = "LLM 调用失败：%s" % e
    else:
        err = "未配置 API key —— 仅返回检索结果"
    degraded = False
    if not answer.strip():
        degraded = True
        answer = local_answer(req.question, ret, err or "生成层不可用")
    materials = set((p.get("law") or "", P.norm_artno(p.get("label"))) for p in ret["provisions"])
    ver = P.verify_citations(answer, materials, State.law_index) if answer else None
    analysis = None
    if req.detailed:
        analysis = P.normalize_analysis(P.extract_json(answer))
    return {"ok": True, "answer": answer, "error": err, "degraded": degraded,
            "evidence": ret, "verify": ver, "analysis": analysis,
            "elapsed": round(time.time() - t0, 2)}


# ---------------------------------------------------------------- 思维链剥离
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.S | re.I)


def strip_think(text):
    """非流式用：删掉 <think>…</think> 块。"""
    if not text:
        return text
    t = _THINK_RE.sub("", text)
    if "</think>" in t:                       # 标签没闭合的兜底
        t = t.split("</think>", 1)[1]
    if "<think>" in t:
        t = t.split("<think>", 1)[0]
    return t.lstrip("\n")


class ThinkStripper(object):
    """流式用：把 <think>…</think> 挡在正文之外。

    MiniMax-M3 这类推理模型把思维链**内联在 delta.content 里**，直接往页面吐
    会滚一片英文自问自答。这里用一个小状态机：在「还判断不出开头是不是 think」
    时先缓冲，确定之后再放行正文。没用 think 的模型（DeepSeek 等）也不受影响。
    """
    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self):
        self.buf = ""
        self.mode = "probe"

    def feed(self, piece):
        if not piece:
            return ""
        self.buf += piece
        while True:
            if self.mode == "probe":
                b = self.buf.lstrip()
                if not b or (len(b) < len(self.OPEN) and self.OPEN.startswith(b)):
                    return ""                     # 再等等才判断得出
                if b.startswith(self.OPEN):
                    self.mode = "thinking"
                    self.buf = b
                    continue
                self.mode = "answer"
                continue
            if self.mode == "thinking":
                i = self.buf.find(self.CLOSE)
                if i < 0:
                    self.buf = self.buf[-len(self.CLOSE):]   # 只看尾巴，别让 buffer 涨
                    return ""
                self.buf = self.buf[i + len(self.CLOSE):].lstrip("\n")
                self.mode = "answer"
                continue
            out, self.buf = self.buf, ""
            return out

    def flush(self):
        """流结束：模型没闭合标签时，至少不要把正文丢掉。"""
        rest, self.buf = self.buf, ""
        if self.mode == "thinking":
            i = rest.find(self.CLOSE)
            self.mode = "answer"
            return rest[i + len(self.CLOSE):].lstrip("\n") if i >= 0 else ""
        self.mode = "answer"
        return rest.lstrip() if rest else ""


def _sse(event, data):
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(data, ensure_ascii=False))


@app.post("/api/ask_stream")
async def api_ask_stream(req: AskReq):
    if State.kb is None:
        async def err_gen():
            yield _sse("fatal", {"error": State.error or "知识库未加载"})
        return StreamingResponse(err_gen(), media_type="text/event-stream")

    async def gen():
        t0 = time.time()
        ret = None

        def final_event(text):
            """最终文本的出口。

            普通模式 → delta（流式正文直接吐给页面）
            详细模式 → analysis（结构化 JSON 面板；解析失败时 raw 兜底）

            两种模式**分开走**，是为了避免详细模式把满屏 JSON 直接怼到用户脸上 ——
            JSON 一定要经过解析、规整，再由前端渲染成分析面板。
            """
            if req.detailed:
                analysis = P.normalize_analysis(P.extract_json(text))
                return _sse("analysis", {"analysis": analysis, "raw": text})
            return _sse("delta", {"t": text})

        try:
            yield _sse("stage", {"stage": "retrieve", "msg": "正在检索知识库…"})
            hist = P.clip_history(req.history)
            q_retr = P.effective_query(req.question, hist)
            ret = await run_in_threadpool(do_retrieve, req.question, req.domains,
                                          req.n_provisions, req.n_cases, q_retr)
            ret["top_cases"] = _top_cases(ret, req.top_k_cases)
            yield _sse("evidence", ret)

            if not _llm_ready():
                yield _sse("notice", {"level": "warn",
                                      "text": "未配置 API key：已改用本地检索模式作答"
                                              "（在 config.json 填 llm.api_key 可恢复完整推理）"})
                yield final_event(local_answer(req.question, ret, "未配置 API key"))
                yield _sse("done", {"elapsed": round(time.time() - t0, 2), "degraded": True})
                return

            yield _sse("stage", {"stage": "generate", "msg": "正在生成回答…"})
            msgs = P.build_messages(req.question, ret["provisions"],
                                    ret["cases"], ret["user"], history=hist,
                                    detailed=req.detailed)
            buf = []
            strip = ThinkStripper()
            n_shown = 0
            async with _client() as c:
                async with c.stream("POST", _endpoint(), headers=_headers(),
                                    json=_llm_body(msgs, stream=True)) as r:
                    if r.status_code != 200:
                        body = (await r.aread()).decode("utf-8", "replace")
                        # 中转失联 / 额度用尽 / key 过期 —— 不让页面停在红条上，
                        # 退回本地检索模式，至少把命中的法条与案例完整给出。
                        why = "接口返回 HTTP %d" % r.status_code
                        yield _sse("notice", {"level": "warn",
                                              "text": "外接模型调用失败（%s），已改用本地检索模式作答"
                                                      % why})
                        yield final_event(local_answer(req.question, ret, why))
                        yield _sse("done", {"elapsed": round(time.time() - t0, 2),
                                            "degraded": True, "llm_error": body[:200]})
                        return
                    async for line in r.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except ValueError:
                            continue
                        ch = obj.get("choices") or []
                        if not ch:
                            continue
                        dl = ch[0].get("delta") or {}
                        # 思维链：开了 reasoning_split 的服务商会把它放在独立字段，
                        # 不在正文里，直接丢弃即可；没开的话它内联在 content 里，
                        # 交给 ThinkStripper 剥离。两边都要处理。
                        if dl.get("reasoning_content"):
                            continue
                        piece = dl.get("content")
                        if piece:
                            buf.append(piece)
                            vis = strip.feed(piece)
                            if vis:
                                if req.detailed:
                                    # 详细模式：JSON 原文不能直接吐给用户（满屏大括号），
                                    # 只报进度，正文等解析成面板再渲染。
                                    n_shown += len(vis)
                                    yield _sse("progress", {"chars": n_shown})
                                else:
                                    yield _sse("delta", {"t": vis})

            tail = strip.flush()
            if tail:
                if req.detailed:
                    n_shown += len(tail)
                    yield _sse("progress", {"chars": n_shown})
                else:
                    yield _sse("delta", {"t": tail})
            answer = strip_think("".join(buf))
            if not answer.strip():                    # 模型返回空正文（偶发）也退回检索模式
                yield _sse("notice", {"level": "warn",
                                      "text": "模型未返回正文，已改用本地检索模式作答"})
                answer = local_answer(req.question, ret, "模型返回空内容")
                yield final_event(answer)
            else:
                yield final_event(answer)
            yield _sse("stage", {"stage": "verify", "msg": "正在核验引用…"})
            materials = set((p.get("law") or "", P.norm_artno(p.get("label")))
                            for p in ret["provisions"])
            ver = P.verify_citations(answer, materials, State.law_index)
            yield _sse("verify", ver)
            yield _sse("done", {"elapsed": round(time.time() - t0, 2),
                                "n_chars": len(answer)})
        except Exception as e:                                     # noqa: BLE001
            log("[err] %s" % traceback.format_exc())
            # 已经检索到材料的话，别让页面只留一条红字 —— 退回本地检索模式
            if ret is not None:
                why = "%s: %s" % (type(e).__name__, e)
                yield _sse("notice", {"level": "warn",
                                      "text": "生成环节异常（%s），已改用本地检索模式作答" % why})
                yield final_event(local_answer(req.question, ret, why))
                yield _sse("done", {"elapsed": round(time.time() - t0, 2), "degraded": True})
                return
            yield _sse("fatal", {"error": "%s: %s" % (type(e).__name__, e)})
            yield _sse("done", {"elapsed": round(time.time() - t0, 2)})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# ======================================================================= 工具
class UploadReq(BaseModel):
    title: str
    text: str
    domain: str = "general"
    kind: str = "case"
    law: str = ""
    label: str = ""


@app.get("/api/uploads")
def api_uploads():
    if State.kb is None:
        return {"ok": False, "items": []}
    items = []
    for i, it in enumerate(State.kb.user_items):
        row = dict(it)
        row["_i"] = i
        row["_vec"] = bool(State.kb.user_vecs is not None)
        items.append(row)
    return {"ok": True, "items": items,
            "embedder_available": bool(State.emb and State.emb.available)}


@app.post("/api/upload")
async def api_upload(req: UploadReq):
    if State.kb is None:
        return JSONResponse({"ok": False, "error": "知识库未加载"}, 500)
    title = (req.title or "").strip()
    text = (req.text or "").strip()
    if not title or not text:
        return JSONResponse({"ok": False, "error": "标题与正文都不能为空"}, 400)
    if len(text) < 10:
        return JSONResponse({"ok": False, "error": "正文太短（至少 10 字）"}, 400)

    item = {"title": title[:120], "text": text[:20000],
            "domain": req.domain or "general", "kind": req.kind or "case",
            "law": (req.law or "").strip(), "label": (req.label or "").strip(),
            "added_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    vec = None
    note = "已入库（向量索引）"
    if State.emb is not None and State.emb.available:
        doc = title + "\n" + text
        try:
            vec = await run_in_threadpool(State.emb.encode_docs, [doc])
            vec = vec[0]
        except Exception as e:                                     # noqa: BLE001
            note = "已入库（编码失败，仅词面索引）：%s" % e
    else:
        note = "已入库（当前为降级模式：仅有词面索引，未生成向量）"
    n = State.kb.add_user_item(item, vec)
    log("[upload] %s（%s）→ 累计 %d 条" % (title[:30], req.kind, n))
    return {"ok": True, "n_user": n, "note": note,
            "vectorized": vec is not None}


@app.delete("/api/uploads/{idx}")
def api_delete_upload(idx: int):
    if State.kb is None:
        return {"ok": False}
    ok = State.kb.delete_user_item(idx)
    return {"ok": ok, "n_user": len(State.kb.user_items)}


@app.get("/api/health")
def health():
    return {"ok": State.kb is not None,
            "embedder": bool(State.emb and State.emb.available),
            "llm": _llm_ready()}


if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
