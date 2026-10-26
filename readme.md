# 法律领域智能体 · 离线演示系统

检索增强的法律问答系统。**向量库完整迁移到本地，生成走外接大模型 API**，
所以现场可以问任何法律问题（不限于题库），并且带引用核验。

---

## 1. 这套东西解决什么问题

答辩现场的限制是：**不能连公司内网、不能保证 GPU、导师可能随口问题库外的问题**。

| 约束 | 本方案怎么解 |
|---|---|
| 连不上公司服务器 | 知识库已导出成普通文件，随 U 盘带走 |
| 本机没有 GPU | 生成交给外接 API；本地只跑 0.6B 的检索编码器（CPU 足够） |
| 导师问题库外的题 | 检索 + 生成，不是回放预制答案，**任何问题都能现场推理** |
| 导师连续追问 | 支持**多轮对话**：追问自动承接上文（检索查询拼接上一轮问题 + 历史进提示词），路由可随追问切换法域 |
| 怕大模型编法条 | 每条回答都过**引用核验**，把《法名》第X条逐条对回材料 |
| 想演示"能学习" | 支持**现场上传**案例进库，上传后立刻能被检索到 |

## 2. 目录结构

```
law-rag/
├─ deploy.bat                ★ 新机一键装环境（自动找 Python，装完自动自检）
├─ start.bat                 ★ 一键启动（15 秒后自动打开浏览器）
├─ doctor.bat / doctor.py    ★ 体检：文件 MD5、venv、依赖、端口、服务、生成接口
├─ pack.bat                  ★ 打包到 U 盘（排除 .venv，拷完自动校验）
├─ findpy.bat                内部工具：按 .venv → py -3 → python → WorkBuddy 内置 找解释器
├─ setup.bat                 旧入口，等同 deploy.bat
├─ run.py                    启动入口（读 config.json 的 host/port）
├─ config.json               ← 填 API key 和模型的地方
├─ app.py                    FastAPI 服务：/api/ask_stream /api/upload /api/stats
├─ retrieval.py              检索引擎（纯 numpy：向量 + 词面锚定融合 + 域软偏向）
├─ embedder.py               Qwen3-Embedding-0.6B 封装（查询侧加 Instruct 前缀）
├─ domain_router.py          L1 法域路由器（论文同款逻辑回归头，CPU 推理）
├─ prompt.py                 提示词 + 引用核验（四态）
├─ selftest.py               环境自检（知识库/编码器/检索/路由一次跑完）
├─ probe_llm.py              探测本机可用的模型端点（免得到现场才发现 key 失效）
├─ import_key.py             把探测到的端点写进 config.json，不用手抄 key
├─ web/index.html            前端（单文件，无 CDN 依赖，断网也能开）
├─ 新电脑部署说明.md          ★ 换电脑/打不开时先看这份
├─ router/                   L1 路由器产物（router_head.pkl + 测试集）
├─ kb/                       ★ 知识库（从服务器导出，可迁移）
│   ├─ provisions.jsonl          法条 72,440 条（含正文）
│   ├─ cases.jsonl               案例 145,339 条（含事实与参考结论）
│   ├─ vectors.f16.npy           217,779 × 1024 维，float16，L2 归一化
│   ├─ manifest.json             条数/维度/md5，用来校验搬运是否损坏
│   └─ user/                     用户上传的增量（自动创建）
└─ models/Qwen3-Embedding-0.6B/   ★ 查询编码器（必须与建库端同一份权重）
```

## 3. 首次安装（联网，5–15 分钟）

```
双击 deploy.bat
```

它做的事：找一个可用的 Python → 建 `.venv` → 装 fastapi/uvicorn/httpx/numpy →
装 torch（CPU 版，约 200MB，最慢）→ 装 sentence-transformers → **自动跑自检与体检**。

> ⚠️ **必须在答辩前、在有网的机器上跑完**。答辩现场没网就没法装。
> 装完之后：**检索层断网可用；生成层仍需联网调外接模型**（断网时会自动降级为
> 「本地检索模式」，给出法条原文 + 相似案例，不会白屏 —— 详见 `新电脑部署说明.md`）。
> 若 `.venv` 是从别的电脑拷来的，`deploy.bat` 会检测到并自动删除重建。

**Python 版本**：3.10 及以上。装 Python 时记得勾选 *Add Python to PATH*。

## 4. 配置生成模型

> 从 GitHub 克隆下来的话，仓库里**不含 `config.json`**（它含真实 API key，已被 `.gitignore` 排除）。
> 先复制模板再填 key：
>
> ```
> copy config.example.json config.json      # Windows
> cp config.example.json config.json        # macOS / Linux
> ```

打开 `config.json`，把 `llm.api_key` 填上：

```json
"llm": {
  "base_url": "https://api.deepseek.com/v1",
  "api_key": "sk-你的key",
  "model": "deepseek-chat"
}
```

常见服务商（都是 OpenAI 兼容接口，只改 `base_url` 和 `model`）：

| 服务商 | base_url | model 示例 |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| 智谱 GLM | `https://open.bigmodel.cn/api/paas/v4` | `glm-4-plus` |
| 阿里通义 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
| 月之暗面 | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |

也可以不改文件，用环境变量覆盖：

```
set LAW_RAG_API_KEY=sk-xxx
set LAW_RAG_BASE_URL=https://api.deepseek.com/v1
set LAW_RAG_MODEL=deepseek-chat
```

> 没填 key 也能启动 —— 此时只跑检索，把命中的法条和相似案例摊开。
> 检索层本身就是可验证的，生成层是锦上添花。

**填完先验一遍端点**（免得现场才发现 key 过期）：

```
.venv\Scripts\python.exe probe_llm.py
```

它会把本机 `~/.codebuddy*` 里登记过的端点逐个试一遍，打印哪些能用、
哪些报 401/404。如果这台机器上装过 CodeBuddy 并配过模型，
`import_key.py` 可以直接把某一条写进 `config.json`，不用手抄 key。

> ⚠️ **外传前记得清空 `config.json` 里的 `api_key`** —— 那是你本机的真实密钥。
> 仓库里只提交 `config.example.json`（空 key），`config.json` 已被 `.gitignore` 排除，不会误传。

## 5. 启动

```
双击 start.bat
```

browser 打开 **http://127.0.0.1:8760** 。首次启动要加载 425MB 向量 + 解析 30 万行文本，
大约 15–40 秒（实测本机 8.6s 就绪），看到右上角出现「法条 72,440 / 案例 145,339 / 编码器 cpu」就是好了。

> 本机 CPU 编码速度实测约 **0.03 s/token**：现场提问都是短问句（30–50 字），
> 单次检索约 1–2 秒，体感没问题；但如果有人粘贴一整篇判决书来问，会等十几秒。
> 这是 CPU 环境的客观限制，别慌。

## 6. 搬到答辩电脑

### 6.1 从 GitHub 克隆（推荐）

知识库、编码器权重等大文件都走 **Git LFS**（`kb/` 与 `models/`），克隆前需先装 LFS：

```
git lfs install
git clone https://github.com/LeslieDian/falv-.git
cd falv-
copy config.example.json config.json   # 然后填 llm.api_key
```

> 如果 `git lfs` 没装，克隆下来的大文件会变成**指针文本**（几十字节），
> 表现为「知识库读不出来 / 编码器模型不存在」。补救：`git lfs install && git lfs pull`。

### 6.2 用 U 盘拷

推荐直接双击 **`pack.bat`**：填一个目标目录（如 `E:\law-rag`），它会拷好并**自动做 MD5 校验**，
还会自动排除 `.venv`、`__pycache__`、临时文件。

手动拷的话，需要一起拷的（实测总计 **≈1.94 GB**：`kb/` 792MB + `models/` 1.15GB + 其余 6MB）：

```
law-rag/            整个目录（含 kb/ 和 models/，但 .venv 一定不要拷）
```

到新电脑后：

1. 双击 `deploy.bat` 装环境（**这一步必须提前做，需要网**）
2. 打开 `config.json` 确认 key（拷过来就带着，不用重填）
3. 双击 `start.bat`，等它自动打开浏览器（或手输 `http://127.0.0.1:8760`）

**校验搬运是否损坏**：双击 `doctor.bat`（会自动比对 `kb/manifest.json` 里的 MD5），
或者手动：

```
.venv\Scripts\python.exe -c "import hashlib,json,os;m=json.load(open('kb/manifest.json',encoding='utf-8'));[print(n, hashlib.md5(open(os.path.join('kb',n),'rb').read()).hexdigest()==v['md5']) for n,v in m['files'].items()]"
```

三个都打印 `True` 就没问题。

## 7. 现场演示流程（建议 5 步）

1. **随手问一个生活化问题**（题库里绝对没有的），比如
   「快递运输中把我的显示器摔坏了，卖家说签收前风险由我承担，谁该赔？」
   → 观察：法域路由 → 检索 → 生成 → 核验 四步依次点亮。
2. **指着「L1 法域路由」讲论文里的路由**：这一条不是写死的，是论文训练的
   逻辑回归头现场推理出来的——比如问「仲裁裁决不服去法院起诉的期限」，
   它会判成「程序 97.2%」并列出四个法域的概率分布，然后按域偏向检索。
3. **展开「检索证据」**：证明答案不是凭空来的，是 21.7 万条向量里检索出来的，
   每条都标了相似度（cos）、词面分、锚定加分。
4. **指着「引用核验」讲防幻觉**：回答里的每个《法名》第X条都逐条比对过，
   `落实 / 出上下文 / 疑似编造` 三态分明。
5. **当场上传一个案例**（右侧面板），传完再问一个相关问题 →
   新内容出现在「命中的自建知识」里。这一步证明系统**可扩展、能学习**。
6. **切法域做对比**：法域选「刑事」再问同一个问题，看检索结果如何收窄。

## 8. 常见故障

**先看 `新电脑部署说明.md` 的故障对照表（更全）。** 核心几条：

| 现象 | 原因 / 处理 |
|---|---|
| 浏览器「127.0.0.1 拒绝连接」 | 服务没启动（或还在启动中）。先 `start.bat`，等 10–40 秒再刷新 |
| `start.bat` 说"还没有运行环境" | 新机没跑过 `deploy.bat` |
| `start.bat` 说".venv 起不来" | 把旧机的 `.venv` 拷过来了，跨机必失效 → 跑 `deploy.bat` 自动重建 |
| 页面**黄色**提示"当前为本地检索模式" | 外接模型没连上（断网/中转挂/额度尽）。检索结果仍完整，不是崩溃 |
| 右上角「降级：仅关键词检索」 | 编码器没加载。检查 `models/Qwen3-Embedding-0.6B/` 是否完整（应有 `model.safetensors` 1.14GB） |
| 「未配置 API key」 | `config.json` 的 `llm.api_key` 为空，或环境变量没生效 |
| `LLM HTTP 401` | key 错 / 过期 / base_url 不对 |
| `LLM HTTP 404` | base_url 少了或多了 `/v1`，或路径不是 `/chat/completions` |
| 启动很慢 | 正常，首次加载要读 425MB 向量 + 30 万行 jsonl |
| 端口被占用 | 改 `config.json` 里 `server.port` |
| 说不清哪里坏了 | 双击 `doctor.bat`，它会直接给结论 |

## 9. 关于"智能"的边界

- 检索质量取决于 **Qwen3-Embedding-0.6B**，不是生成模型。换生成模型不影响召回。
- **不要**试图用第三方 embedding API 替换查询编码器 —— 那与库内向量不在同一空间，
  检索会静默变差（不报错，只是不准）。要换就得整库重跑。
- 案例库是"参考裁判思路"用的，法律上不构成判例约束，别把案例事实当本案事实说。
- **这套本地演示跑的是「检索 + L1 路由 + 外接生成」**，没有加载论文里那套 LoRA / MoE
  门控（它们要 8B 基座 + A800）。所以别说"这是我的模型答的"——准确说法是
  「法条与案例来自我的知识库，路由头来自我训练的逻辑回归，生成交给外接大模型，
  且生成被强制约束在检索材料内并通过引用核验」。详细口径见 `答辩口径.md`。

## 10. 检索层的两个质量修正（2026-09-28）

语料逐字节校验通过之后，又按"检索用途"做了一遍质量扫描，查出两个会直接拉低回答质量的问题，
已在 `retrieval.py` 里修掉（**没有改动任何数据文件，向量矩阵也没重建**）：

**① top-k 名额被副本吃掉。** 库内法条有 14,456 行冗余副本（20%：6,047 组纯重复 +
6,971 组是同一部法规的不同版本），案例约 24% 在不同集合重复收录。
实测某题 top-8 里 8 条其实只有 3 条，模型还会看到"同一条号、两段不同正文"的自相矛盾材料。
现在排序后按「同一法条 / 同一案情」折叠，名额让给后面真正不同的候选。

**② 七成案例对"相似案例对比"没有价值。** 案例来自 9 个指令数据集，参考答案分三类：

| 可用 | 一般 | 无用（标注型任务） |
|---|---|---|
| `judgement_predit` 判决推理、`sent_pred` 后果推断、`op_sum` 摘要 | `jud_doc_sum` 文书摘要、`leg_case_cls` 风险分类 | `jud_read_compre` 阅读理解（答案是日期/人名）、`leg_ele_extra`（答案是 `O O B-purpose`）、`leg_eve_detec`（答案是 `标签:'婚后有子女'`）、`sim_case_match`（答案是 `{"LABEL":"C"}`） |

现在按可用性分档惩罚/加权（标注型 -0.60 近乎排除、判决推理型 +0.03），
案例候选池从 400 扩到 1,500 保证惩罚后仍能凑满 6 条。标注型案例的"参考答案"
会换成一句说明，不会把标签串当结论喂给模型。

**效果**：返回法条 64/64 全部唯一（修前 54/64）；返回案例任务类型 91.7% 为判决/摘要类；
12 题生僻问题回归 **12/12 命中、引用核验疑似编造 0**；检索排序耗时 27 ms
（用户可感知总耗时约 1.0–1.2 s）。完整证据见 `语料核验报告.md`。

