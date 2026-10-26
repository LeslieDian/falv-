# -*- coding: utf-8 -*-
"""提示词组装 + 引用核验（论文里 verify 工具的本地复刻）。

为什么即使换成外接 API 也要保留核验层
------------------------------------
外接大模型比本地 8B 更强，但**更容易编法条** —— 它见过太多法条，会说得很像真的。
所以「生成」和「核查」必须分开：生成给出答案，核验逐条比对答案里的《法名》第X条
是否真的来自本次检索到的材料。三态输出：

    found          材料里确有该条          → 可信
    out_of_ctx     库里有，但不在本次材料里  → 出上下文（可能是模型自己"想"起来的）
    missing        库里也找不到            → **疑似编造**，标红
    law_unresolved 法名解析不出来           → 无法核验（简称/非规范名）

这套东西让导师现场追问「你凭什么信它没编」时有东西可看 —— 这正是本地模型
换成 API 后唯一不能丢的部分。
"""
import json
import re
from collections import defaultdict

SYSTEM_PROMPT = """你是「法律领域智能体」的回答模块。请严格依据下面提供的【法条材料】与【相似案例】回答用户的法律问题。

硬性规则：
1. 引用法条必须写成《法律全称》第X条 的格式，且**只能引用【法条材料】里真实出现过的条文**。材料里没有的不要引用。
2. 【相似案例】只用于参考裁判思路，**不得把案例中的当事人、金额、事实当作本案事实**。案例事实里出现案号的可注明案号，没有就写「某案例」，**绝不编造案号、裁判结果或生效日期**。
3. 如果关键事实缺失，不要硬下结论 —— 明确列出「需要补充的信息」（如鉴定意见、责任认定书、合同条款），并按不同事实版本分情形给结论。
4. 使用简体中文，条理清晰。不要输出与问题无关的免责声明或致歉。
5. 对话可能是多轮的：【对话历史】仅供参考（理解指代、承接前文事实）；本轮要回答的永远是【用户问题】。若用户在追问中补充了新事实（如「他还逃逸了」「血醇含量是 95」），以新事实为准重新分析，不要照搬上一轮结论。历史中上一轮的回答不是依据，引用仍只能出自本轮【法条材料】。

回答结构（按此顺序输出，用 markdown 二级标题）：

## 一、事实梳理与待确认信息
先复述可以确认的事实；再列出影响定性的关键未知信息（没有就写「无明显待确认事实」）。

## 二、法律依据
逐条列出适用的条文：《法律全称》第X条 —— 引用条文原文关键句，并说明**该条的要件如何对应到本案事实**（要件 ↔ 事实逐点对应）。不许引用材料外的条文。

## 三、相似案例对比
从【相似案例】中挑 2～3 个最相关的，逐个对比：
- 案例事实要点（一两句）
- 其参考结论 / 裁判思路
- **与本案的异同**：哪些情节相同因此可参照，哪些关键情节不同因此只能参考不能照搬。
若【相似案例】中没有可比的，直接写明「检索到的案例与本案差异较大，不宜直接参照」，不要硬凑。

## 四、结论（分情形）
用「可能构成 / 倾向于构成 / 需证据进一步确认」的措辞，按不同事实版本分别给结论；能确定的部分就明确说。

## 五、补充建议
举证责任、证据收集、时效或程序要点（视需要，两三点即可；确实无关可省略）。
"""

# ===================================================================== 详细分析模式
# 为什么要有第二种输出形态
# ----------------------
# 默认回答是「结论导向」的段落式，适合直接读结论。但答辩现场导师常问：
# 「你这个结论是怎么推出来的？要件一���二为什么成立？哪个事实是决定性的？」
# 只有一个结论答不上来。详细分析模式把推理**显式摊开**：事实→争点→要件拆解
# →逐条论证→证据→结论，每一步都能对照材料检查。
#
# 关键设计：详细模式**不是让模型自由发挥**，而是要求它输出**结构化 JSON**，
# 再由前端渲染成面板。这样做有三个好处：
#   1. 字段固定 → 前端能稳定渲染，模型偶尔多写少写字段也不会崩版；
#   2. 每一步都带「依据」（法条/案例），可以逐条核验 → 幻觉更容易被发现；
#   3. 模型输出 JSON 时不会自由插入「免责声明」之类的水词，篇幅更紧凑。
#
# 若模型仍返回了 markdown（不听话），前端会回退到「按正文渲染」，
# 不会白屏 —— 这一点由 web/index.html 的 renderAnalysis 兜底。

ANALYSIS_SCHEMA_DOC = """{
  "case_summary": "一句话概括案情（不含结论）",
  "facts": ["已确认的事实", "..."],
  "unknowns": ["影响定性的待确认事实", "..."],
  "issues": [
    {
      "title": "争点一名称",
      "answer": "该争点的结论倾向（一句话）",
      "elements": [
        {
          "element": "构成要件/法律要件",
          "basis": "《法律全称》第X条",
          "reasoning": "要件如何落到本案事实上",
          "status": "成立 | 不成立 | 待确认"
        }
      ],
      "counter": "对方可能主张及其反驳（如有）"
    }
  ],
  "evidence": ["需要收集的证据", "..."],
  "conclusion": "最终结论（分情形，标明倾向性用语）",
  "uncertainty": "本结论的边界与风险"
}"""


def build_analysis_system_prompt():
    """详细分析模式的系统提示词。

    与默认模式的差别只在**输出形态**：硬性规则（只能引用给定材料、
    不许编造案例、缺事实要列待确认）完全保留 —— 这是防幻觉的底线，
    换格式不能把底线也换了。
    """
    rules = SYSTEM_PROMPT.split("回答结构（按此顺序输出，用 markdown 二级标题）：")[0]
    return (
        rules
        + """回答结构（**必须**只输出一个 JSON 对象，不要输出任何解释性文字、不要用 markdown 代码块包裹）：

```
"""
        + ANALYSIS_SCHEMA_DOC
        + """
```

字段说明：
- `case_summary`：一句话案情，不含结论。
- `facts`：已确认的事实，逐条列。
- `unknowns`：影响定性的待确认事实（缺失就写「无明显待确认事实」）。
- `issues`：**逐个法律争点**展开。`elements` 里每个要件都必须写 `basis`
  （只能引用【法条材料】里出现过的条文，写不满就少写，不要硬凑）、
  `reasoning`（要件 ↔ 本案事实的逐点对应）、`status`（三态之一）。
  这是详细分析的核心 —— 模型必须把「怎么推出来的」写出来。
- `evidence`：需要补充的证据 / 举证要点。
- `conclusion`：最终结论，按不同事实版本分情形，措辞用「可能构成 / 倾向于构成 / 需证据进一步确认」。
- `uncertainty`：结论的边界、可能推翻结论的情形。

补充要求：
1. JSON 必须是合法的、用双引号包裹的字符串，不能有注释、不能有尾逗号。
2. 字符串内不要出现未转义的换行（用「；」分隔多条内容）。
3. 只输出 JSON，不要前后加任何说明文字。
"""
    )


# ===================================================================== 提示词
def clip_history(history, n_msgs=4, n_chars=1200):
    """清洗前端传来的多轮历史：只留 user/assistant、逐条截断、最多 n_msgs 条。"""
    out = []
    for m in history or []:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content[:n_chars]})
    return out[-n_msgs:]


# 材料预算（控制 prompt 长度）
PROV_TEXT_MAX = 700
CASE_FACT_MAX = 500
CASE_ANS_MAX = 260

LAWNAME_RE = re.compile(r"《([^》]{2,60})》")
ARTNO_RE = re.compile(r"第([一二三四五六七八九十百零〇0-9]{1,6})条")


# ===================================================================== 法名索引
def alias_of(law):
    """粗粒度简称：去国名前缀、去两高前缀、取书名号内的主体法。"""
    a = law or ""
    m = re.search(r"《([^》]+)》", a)
    if m:
        a = m.group(1)
    a = a.replace("中华人民共和国", "")
    a = re.sub(r"^(最高人民法院|最高人民检察院)(关于)?", "", a)
    a = re.sub(r"^关于", "", a)
    return a.strip()


class LawIndex(object):
    """法名 → 该法在库中的条号集合。用于区分 out_of_ctx 与 missing。"""

    def __init__(self, provisions):
        self.full = defaultdict(set)      # 全称 → {条号}
        self.articles_by_uid = {}         # (法名, 条号) → provision dict
        for p in provisions:
            law = (p.get("law") or "").strip()
            label = (p.get("label") or "").strip()
            no = norm_artno(label)
            if not law:
                continue
            self.full[law].add(no)
            self.articles_by_uid[(law, no)] = p
        # 简称表：只在唯一映射时启用，避免歧义
        bucket = defaultdict(set)
        for law in self.full:
            bucket[alias_of(law)].add(law)
        self.alias = {k: list(v)[0] for k, v in bucket.items() if len(v) == 1 and k}

    def resolve(self, name):
        """把回答里写的法名解析成库内全称。"""
        name = (name or "").strip()
        if not name:
            return None
        if name in self.full:
            return name
        if name in self.alias:
            return self.alias[name]
        # 去掉《》里常见修饰后再试
        for cand in (name.replace(" ", ""), "中华人民共和国" + name):
            if cand in self.full:
                return cand
        # 包含匹配（唯一命中才算）
        hits = [law for law in self.full if name in law or law in name]
        if len(hits) == 1:
            return hits[0]
        return None

    def exists(self, law, artno):
        return artno is not None and artno in self.full.get(law, set())


CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def norm_artno(s):
    """'第十七条' / '第17条' / '十七' / 17 → '17'（字符串）。认不出来返回原串。"""
    if s is None:
        return None
    if isinstance(s, int):
        return str(s)
    t = str(s).strip()
    m = ARTNO_RE.search(t)
    if m:
        t = m.group(1)
    if t.isdigit():
        return str(int(t))
    # 中文数字
    if not t or any(ch not in CN_DIGITS and ch not in "十百千" for ch in t):
        return None
    total, section, num = 0, 0, 0
    for ch in t:
        if ch in CN_DIGITS:
            num = CN_DIGITS[ch]
        elif ch == "十":
            section += (num or 1) * 10
            num = 0
        elif ch == "百":
            section += (num or 1) * 100
            num = 0
        elif ch == "千":
            section += (num or 1) * 1000
            num = 0
    total = section + num
    return str(total) if total else None


# ===================================================================== 提示词
def clip_history(history, n_msgs=4, n_chars=1200):
    """清洗前端传来的多轮历史：只留 user/assistant、逐条截断、最多 n_msgs 条。"""
    out = []
    for m in history or []:
        role = (m.get("role") or "").strip()
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content[:n_chars]})
    return out[-n_msgs:]


def effective_query(question, history):
    """多轮检索查询：把上一轮用户问题拼进本轮问题，解决指代/省略（「那醉驾呢」）。

    只拼 user 轮（assistant 回答太长且会污染检索空间），每段截断防超长。
    """
    prev = ""
    for m in history or []:
        if m.get("role") == "user" and (m.get("content") or "").strip():
            prev = m["content"].strip()
    prev = prev[:300]
    if not prev or prev == question:
        return question
    return (prev + "\n" + question.strip())[:800]


def build_messages(question, provisions, cases, user_items=None, history=None,
                   detailed=False):
    """组装发给模型的消息。

    detailed=False → 默认模式：markdown 段落式回答
    detailed=True  → 详细分析模式：要求模型输出结构化 JSON（见 ANALYSIS_SCHEMA_DOC）
    """
    lines = ["【用户问题】", question.strip(), ""]

    if history:
        lines.append("【对话历史】（仅供理解上下文，不是本轮回答依据）")
        for m in history:
            tag = "用户" if m["role"] == "user" else "助手（上一轮回答）"
            lines.append("- %s：%s" % (tag, m["content"].replace("\n", " ")))
        lines.append("")

    lines.append("【法条材料】共 %d 条（按相关度排序）" % len(provisions))
    if provisions:
        for i, p in enumerate(provisions, 1):
            head = "《%s》%s" % (p.get("law") or "", p.get("label") or "")
            ch = p.get("chapter") or ""
            st = p.get("status") or ""
            meta = "（%s%s%s）" % (ch, "，" if ch and st else "", st) if (ch or st) else ""
            txt = (p.get("text") or "").strip()[:PROV_TEXT_MAX]
            lines.append("%d. %s%s\n   %s" % (i, head, meta, txt))
    else:
        lines.append("（无）")
    lines.append("")

    if user_items:
        lines.append("【用户自行上传的知识】共 %d 条" % len(user_items))
        for i, u in enumerate(user_items, 1):
            lines.append("%d. %s\n   %s" % (
                i, u.get("title") or "", (u.get("text") or "").strip()[:PROV_TEXT_MAX]))
        lines.append("")

    lines.append("【相似案例】共 %d 条（参考裁判思路用）" % len(cases))
    if cases:
        for i, c in enumerate(cases, 1):
            dom = c.get("domain") or ""
            task = c.get("task") or ""
            fact = (c.get("fact") or "").strip()[:CASE_FACT_MAX]
            ans = (c.get("answer") or "").strip()[:CASE_ANS_MAX]
            lines.append("%d. 〔%s%s〕\n   事实：%s\n   参考结论：%s"
                         % (i, dom, ("／" + task) if task else "", fact, ans))
    else:
        lines.append("（无）")
    lines.append("")
    if detailed:
        lines.append("请依据上述材料回答【用户问题】，并按系统消息中规定的 JSON 格式输出详细分析。")
    else:
        lines.append("请依据上述材料回答【用户问题】。")
    system = build_analysis_system_prompt() if detailed else SYSTEM_PROMPT
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n".join(lines)}]


# ===================================================================== 详细分析 JSON
def extract_json(text):
    """从模型输出里抠出 JSON 对象。

    模型有时不听话，会输出 ```json 包裹、带前后废话、或用全角引号。
    这里逐层兜底，尽量把它救回来；实在救不回来就返回 None，
    由调用方回退到「按正文渲染」—— 宁可少个面板，也不能让回答消失。
    """
    if not text:
        return None
    s = text.strip()
    # 1) 去掉 markdown 代码块包裹
    if "```" in s:
        parts = s.split("```")
        for p in parts[1:]:
            p = p.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                cand = _try_json(p)
                if cand is not None:
                    return cand
        s = " ".join(parts)
    # 2) 找第一个 { 到最后一个 } 之间的内容
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        cand = _try_json(s[i:j + 1])
        if cand is not None:
            return cand
    # 3) 全文就是 JSON
    return _try_json(s)


def _try_json(s):
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except ValueError:
        pass
    # 兜底：全角引号 → 半角；去掉尾逗号
    t = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    t = re.sub(r",\s*([}\]])", r"\1", t)
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except ValueError:
        return None


def normalize_analysis(obj):
    """把模型给的 JSON 规整成前端要的形状（缺字段补空，类型纠偏）。"""
    if not isinstance(obj, dict):
        return None

    def _str(v):
        if v is None:
            return ""
        if isinstance(v, str):
            return v.strip()
        return str(v).strip()

    def _list(v):
        if v is None:
            return []
        if isinstance(v, str):
            return [x.strip() for x in re.split(r"[\n；;]+", v) if x.strip()]
        if isinstance(v, list):
            out = []
            for x in v:
                if isinstance(x, dict):
                    t = _str(x.get("text") or x.get("content") or x.get("value"))
                    if t:
                        out.append(t)
                else:
                    t = _str(x)
                    if t:
                        out.append(t)
            return out
        t = _str(v)
        return [t] if t else []

    issues = []
    for it in (obj.get("issues") or []):
        if not isinstance(it, dict):
            t = _str(it)
            if t:
                issues.append({"title": t, "answer": "", "elements": [], "counter": ""})
            continue
        els = []
        for e in (it.get("elements") or []):
            if not isinstance(e, dict):
                t = _str(e)
                if t:
                    els.append({"element": t, "basis": "", "reasoning": "", "status": ""})
                continue
            els.append({
                "element": _str(e.get("element") or e.get("name")),
                "basis": _str(e.get("basis") or e.get("citation") or e.get("article")),
                "reasoning": _str(e.get("reasoning") or e.get("analysis") or e.get("reason")),
                "status": _str(e.get("status") or e.get("result")),
            })
        issues.append({
            "title": _str(it.get("title") or it.get("issue") or it.get("name")),
            "answer": _str(it.get("answer") or it.get("conclusion") or it.get("view")),
            "elements": els,
            "counter": _str(it.get("counter") or it.get("counterargument") or it.get("rebuttal")),
        })

    return {
        "case_summary": _str(obj.get("case_summary") or obj.get("summary")),
        "facts": _list(obj.get("facts")),
        "unknowns": _list(obj.get("unknowns") or obj.get("missing_facts")),
        "issues": issues,
        "evidence": _list(obj.get("evidence") or obj.get("evidence_needed")),
        "conclusion": _str(obj.get("conclusion")),
        "uncertainty": _str(obj.get("uncertainty") or obj.get("limitations")),
    }


# ===================================================================== 核验
def verify_citations(answer, materials, law_index):
    """→ dict(found=[], out_of_ctx=[], missing=[], law_unresolved=[], n_citations=)

    materials: 本次真正喂给模型的 [(法名, 条号)] 集合
    """
    found, out_of_ctx, missing, unresolved = [], [], [], []
    seen = set()
    for m in LAWNAME_RE.finditer(answer or ""):
        raw_name = m.group(1)
        tail = answer[m.end():m.end() + 12]
        am = ARTNO_RE.search(tail)
        artno = norm_artno(am.group(1)) if am else None
        law = law_index.resolve(raw_name)
        if law is None:
            key = ("?", raw_name, artno)
            if key in seen:
                continue
            seen.add(key)
            unresolved.append({"law": raw_name, "article": artno})
            continue
        key = (law, artno)
        if key in seen:
            continue
        seen.add(key)
        item = {"law": law, "article": artno,
                "label": ("第%s条" % artno) if artno else ""}
        if (law, artno) in materials:
            found.append(item)
        elif artno is None:
            found.append(item)              # 只提法名没提条号，不算错
        elif law_index.exists(law, artno):
            out_of_ctx.append(item)
        else:
            missing.append(item)
    n = len(found) + len(out_of_ctx) + len(missing) + len(unresolved)
    return {"found": found, "out_of_ctx": out_of_ctx,
            "missing": missing, "law_unresolved": unresolved,
            "n_citations": n,
            "verdict": ("ok" if not missing else "suspect")}
