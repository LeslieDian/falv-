# -*- coding: utf-8 -*-
"""新机体检：一键查出「为什么打不开 / 为什么跑不起来」。

用法（在 law-rag 目录下）：
    python doctor.py                 # 检查当前目录
    python doctor.py --root D:\\law-rag   # 检查另一个目录（例如刚拷好的 U 盘内容）
    python doctor.py --quick         # 跳过 MD5 校验（只比大小，几秒出结果）

只依赖 Python 标准库，**任何** Python 3.8+ 都能跑 —— 包括还没装好依赖时。
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

OK, BAD, WARN = "  [OK] ", "  [!!] ", "  [~~] "
FAIL, NOTE = [], []

MODEL_SUBDIR = os.path.join("models", "Qwen3-Embedding-0.6B")
MODELS = {
    "model.safetensors": 1191586416,
    "tokenizer.json": 11423705,
    "config.json": 727,
    "modules.json": 349,
    "vocab.json": 2776833,
    "merges.txt": 1671853,
    "tokenizer_config.json": None,
    "config_sentence_transformers.json": None,
    "generation_config.json": None,
}
SRC = ["app.py", "retrieval.py", "prompt.py", "embedder.py", "domain_router.py",
       "run.py", "selftest.py", "requirements.txt",
       os.path.join("web", "index.html"),
       os.path.join("router", "router_head.pkl")]

# config.json 含真实 API key，不随代码分发（见 .gitignore）。
# 新机器首次使用请先复制 config.example.json → config.json 并填 key。
SRC_OPTIONAL = ["config.json"]


def mb(n):
    return "%.1f MB" % (n / 1048576.0)


def head(t):
    print("\n" + "-" * 68)
    print(t)
    print("-" * 68)


def md5(path, chunk=8 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ [1] 文件
def check_files(root, quick):
    head("[1/6] 数据与代码文件")
    man = os.path.join(root, "kb", "manifest.json")
    if not os.path.exists(man):
        print(BAD + "kb/manifest.json 缺失 —— 知识库没拷全")
        FAIL.append("kb/manifest.json 缺失")
        return
    try:
        with open(man, encoding="utf-8") as f:
            m = json.load(f)
    except Exception as e:                                          # noqa: BLE001
        print(BAD + "kb/manifest.json 读不出来：%s" % e)
        FAIL.append("manifest.json 损坏")
        return
    print("  知识库自称：%s 法条 + %s 案例 = %s 条向量（%s 维 %s）"
          % (m.get("n_provisions"), m.get("n_cases"), m.get("n_total"),
             m.get("dim"), m.get("dtype")))
    print("  建库时间：%s" % m.get("built_at"))

    for name, exp in (m.get("files") or {}).items():
        p = os.path.join(root, "kb", name)
        if not os.path.exists(p):
            print(BAD + "kb/%-22s 缺失" % name)
            FAIL.append("kb/%s 缺失（拷漏了）" % name)
            continue
        got = os.path.getsize(p)
        if got != exp.get("bytes"):
            print(BAD + "kb/%-22s 大小不符：%d ≠ %d —— 拷贝不完整，请重拷"
                  % (name, got, exp.get("bytes")))
            FAIL.append("kb/%s 大小不对（拷贝中断）" % name)
            continue
        if quick:
            print(OK + "kb/%-22s %-10s 大小一致（未校验 MD5）" % (name, mb(got)))
            continue
        h = md5(p)
        if h != exp.get("md5"):
            print(BAD + "kb/%-22s MD5 不符：%s ≠ %s —— 文件损坏，请重拷"
                  % (name, h[:12], str(exp.get("md5"))[:12]))
            FAIL.append("kb/%s 内容校验失败" % name)
        else:
            print(OK + "kb/%-22s %-10s MD5 一致" % (name, mb(got)))

    # 编码器权重
    mdir = os.path.join(root, MODEL_SUBDIR)
    if not os.path.isdir(mdir):
        print(BAD + "%s 目录不存在" % MODEL_SUBDIR)
        FAIL.append("编码器模型目录缺失（检索无法语义化，会退化成关键词匹配）")
    else:
        n_f = 0
        tot = 0
        for r, _d, fs in os.walk(mdir):
            for f in fs:
                n_f += 1
                tot += os.path.getsize(os.path.join(r, f))
        print(OK + "%s  %d 个文件 / %s" % (MODEL_SUBDIR, n_f, mb(tot)))
        for f, exp in MODELS.items():
            p = os.path.join(mdir, f)
            if not os.path.exists(p):
                print(BAD + "  %s 缺失" % f)
                FAIL.append("编码器缺文件 %s" % f)
            elif exp and os.path.getsize(p) != exp:
                print(WARN + "  %s 大小与出厂不符（%d ≠ %d）"
                      % (f, os.path.getsize(p), exp))
                NOTE.append("编码器 %s 大小异常，若检索结果异常请重拷" % f)

    missing = [p for p in SRC if not os.path.exists(os.path.join(root, p))]
    if missing:
        print(BAD + "代码文件缺失：%s" % ", ".join(missing))
        FAIL.append("代码文件缺失：%s" % ", ".join(missing))
    else:
        print(OK + "代码文件齐全（app/retrieval/prompt/embedder/router/web…）")

    no_cfg = [p for p in SRC_OPTIONAL if not os.path.exists(os.path.join(root, p))]
    if no_cfg:
        print(WARN + "config.json 缺失 —— 复制 config.example.json 为 config.json "
                     "并填 llm.api_key 后才能用生成层（检索层不受影响）")
        NOTE.append("未创建 config.json：只能演示检索层")


# ------------------------------------------------------------------ [2] venv
def check_venv(root):
    head("[2/6] Python 运行环境")
    # 优先：自带运行时 runtime\（解压 partC 即得，免安装、免联网）
    rt = os.path.join(root, "runtime", "python.exe")
    if os.path.exists(rt):
        probe = ("import sys,numpy,torch,sentence_transformers,fastapi;"
                 "print(sys.version.split()[0])")
        r = subprocess.run([rt, "-c", probe], capture_output=True, text=True)
        if r.returncode == 0:
            print(OK + "自带运行时 runtime\\ 可用（Python %s）" % r.stdout.strip())
            print("       免安装、免 pip、免联网 —— 本机不需要另装 Python")
            return rt
        print(WARN + "runtime\\ 存在但依赖不全，退回检查 .venv\\")
    py = os.path.join(root, ".venv", "Scripts", "python.exe")
    if not os.path.exists(py):
        print(BAD + "既没有 runtime\\ 也没有 .venv —— 新机上必须先装一次")
        print("       → 双击【deploy.bat】：有 partC 就走免安装，没有就联网装")
        FAIL.append("缺少运行环境：需先跑 deploy.bat")
        return None
    cfg = os.path.join(root, ".venv", "pyvenv.cfg")
    home = ""
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8", errors="replace") as f:
            for ln in f:
                if ln.lower().startswith("home"):
                    home = ln.split("=", 1)[1].strip()
    if home and not os.path.isdir(home):
        print(BAD + ".venv 是从别的电脑拷来的，指向的解释器已不存在：")
        print("       %s" % home)
        print("       → 虚拟环境不能跨机拷贝。请删除 .venv 文件夹，再跑【deploy.bat】")
        FAIL.append(".venv 跨机失效：删掉 .venv 后重建")
        return None
    r = subprocess.run([py, "-c", "import sys;print(sys.version.split()[0])"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(BAD + ".venv 里的 python 起不来：%s" % (r.stderr or "")[:200])
        FAIL.append(".venv 已损坏：删掉后重建")
        return None
    print(OK + ".venv 可用（Python %s，基于 %s）" % (r.stdout.strip(), home or "?"))
    return py


# ------------------------------------------------------------------ [3] 依赖
def check_deps(py):
    head("[3/6] 依赖包")
    if not py:
        print(WARN + "先备好运行时（runtime\\ 或 .venv）才能检查依赖")
        return
    groups = [("Web 框架", "fastapi,uvicorn,httpx,pydantic"),
              ("数值", "numpy"),
              ("向量编码器", "sentence_transformers,torch")]
    for name, mods in groups:
        code = "import " + mods.replace(",", ", ")
        r = subprocess.run([py, "-c", code], capture_output=True, text=True)
        if r.returncode == 0:
            print(OK + name)
        else:
            line = [l for l in (r.stderr or "").strip().splitlines() if l.strip()]
            print(BAD + "%s 缺失：%s" % (name, line[-1][:120] if line else "?"))
            FAIL.append("依赖缺失（%s）→ 重跑 deploy.bat" % name)
    # torch 版本旁证
    r = subprocess.run([py, "-c", "import torch;print(torch.__version__, torch.cuda.is_available())"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print("       torch %s（CUDA %s）—— CPU 版即可，本演示只做查询编码"
              % tuple((r.stdout.strip().split() + ["?"])[:2]))


# ------------------------------------------------------------------ [4] 端口
def check_port(port):
    head("[4/6] 端口 %d" % port)
    s = socket.socket()
    s.settimeout(1.5)
    try:
        s.connect(("127.0.0.1", port))
        print(OK + "有进程在监听 %d —— 服务可能已经在跑，直接开浏览器" % port)
        return True
    except Exception:                                              # noqa: BLE001
        print(WARN + "%d 空闲（没有服务在跑）→ 双击【start.bat】启动" % port)
        return False
    finally:
        s.close()


# ------------------------------------------------------------------ [5] 服务
def check_health(port):
    head("[5/6] 服务健康检查")
    # 必须绕开系统代理：本机回环地址走代理会得到假的 502/拒绝
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with op.open("http://127.0.0.1:%d/api/health" % port, timeout=6) as r:
            j = json.loads(r.read().decode("utf-8", "replace"))
        print(OK + "服务在线：知识库=%s 编码器=%s 生成层=%s"
              % ("就绪" if j.get("ok") else "未就绪",
                 "就绪" if j.get("embedder") else "降级",
                 "已配置" if j.get("llm") else "未配置"))
        if not j.get("ok"):
            FAIL.append("服务在线但知识库没加载成功 —— 看服务窗口的报错")
    except Exception as e:                                         # noqa: BLE001
        print(WARN + "取不到 /api/health（%s）—— 服务没启动，属于正常" % type(e).__name__)


# ------------------------------------------------------------------ [6] 生成层
def check_llm(root):
    head("[6/6] 生成层配置（外接 API）")
    cp = os.path.join(root, "config.json")
    key = url = model = ""
    if os.path.exists(cp):
        try:
            with open(cp, encoding="utf-8") as f:
                cfg = json.load(f).get("llm") or {}
            key = (cfg.get("api_key") or "").strip()
            url = (cfg.get("base_url") or "").strip()
            model = (cfg.get("model") or "").strip()
        except Exception as e:                                     # noqa: BLE001
            print(BAD + "config.json 解析失败：%s" % e)
            FAIL.append("config.json 格式错误")
    if not key:
        print(WARN + "没填 llm.api_key → 系统仍可用，但只输出检索结果（不会写分析结论）")
        NOTE.append("未配置 key：只能演示检索层")
    else:
        print(OK + "已配置：%s（key %s…%s）" % (model, key[:6], key[-4:]))
    if url:
        host, _, rest = url.split("//", 1)[-1].partition("/")
        port = 443
        if ":" in host:
            host, _, ps = host.partition(":")
            port = int(ps) if ps.isdigit() else 443
        s = socket.socket()
        s.settimeout(5)
        try:
            s.connect((host, port))
            print(OK + "生成接口可达：%s:%d" % (host, port))
        except Exception as e:                                     # noqa: BLE001
            print(WARN + "连不上生成接口 %s:%d（%s）" % (host, port, type(e).__name__))
            NOTE.append("生成接口当前不可达 —— 不影响启动与检索，"
                        "系统会自动降级为「本地检索模式」继续作答")
        finally:
            s.close()
        print("       base_url=%s" % url)


# ------------------------------------------------------------------ 汇总
def report(root, port, running=False):
    print("\n" + "=" * 68)
    if not FAIL:
        if running:
            print("  体检通过 —— 服务已经在跑，直接打开： http://127.0.0.1:%d" % port)
        else:
            print("  体检通过 —— 结论：可以启动服务")
            print("  下一步：双击【start.bat】，然后浏览器打开 http://127.0.0.1:%d" % port)
    else:
        print("  发现 %d 个问题：" % len(FAIL))
        for i, f in enumerate(FAIL, 1):
            print("   %d) %s" % (i, f))
        print()
        print("  按这个顺序处理：")
        print("   · 若提示缺运行环境 / 缺依赖 → 双击【deploy.bat】")
        print("       它自带三条路：有 runtime\\ 免安装 → 有 wheels\\ 离线装 → 都没有才联网装")
        print("   · 若提示文件大小或 MD5 不符 → 传输没传完，重新传一份")
        print("   · 若 .venv 是从别的电脑拷来的 → 删掉整个 .venv 文件夹再部署")
    if NOTE:
        print("\n  提醒：")
        for n in NOTE:
            print("   - %s" % n)
    print("=" * 68)
    print("  （本机：%s ；被检查目录：%s）" % (sys.version.split()[0], root))
    return 1 if FAIL else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=HERE)
    ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--quick", action="store_true", help="跳过 MD5 校验")
    a = ap.parse_args()
    root = os.path.abspath(a.root)

    print("=" * 68)
    print("  法律领域智能体 · 新机体检")
    print("  被检查目录：%s" % root)
    print("  解释器：    %s" % sys.executable)

    check_files(root, a.quick)
    py = check_venv(root)
    check_deps(py)
    listen = check_port(a.port)
    check_health(a.port)
    check_llm(root)
    return report(root, a.port, listen)


if __name__ == "__main__":
    sys.exit(main())
