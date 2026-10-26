# -*- coding: utf-8 -*-
"""从国内镜像下载查询编码器 Qwen3-Embedding-0.6B（约 1.15 GB）。

用途：新电脑上不想/不能传输 models/ 那 900MB 压缩包时，改跑本脚本自己下。
     下载完 deploy.bat / selftest.py 就能正常识别编码器。

用法（在 law-rag 目录下）：
    .venv\\Scripts\\python.exe -X utf8 fetch_model.py
    或（还没建虚拟环境时用任意 Python）：
    python -X utf8 fetch_model.py

可换镜像：set LAW_RAG_HF_BASE=https://hf-mirror.com
本脚本只依赖标准库；支持断点续传（中断后重跑会接着下）。
"""
import os
import sys
import time
import urllib.error
import urllib.request

REPO = 'Qwen/Qwen3-Embedding-0.6B'
FILES = [
    'config.json',
    'config_sentence_transformers.json',
    'generation_config.json',
    'modules.json',
    'tokenizer_config.json',
    'merges.txt',
    'vocab.json',
    'tokenizer.json',
    'model.safetensors',
    '1_Pooling/config.json',
]
BASES = [b.rstrip('/') for b in [
    os.environ.get('LAW_RAG_HF_BASE', ''),
    'https://hf-mirror.com',
    'https://huggingface.co',
] if b.strip()]

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, 'models', 'Qwen3-Embedding-0.6B')


def fetch(base, rel, dest):
    url = '%s/%s/resolve/main/%s' % (base, REPO, rel)
    tmp = dest + '.part'
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    pos = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url)
    if pos:
        req.add_header('Range', 'bytes=%d-' % pos)
    with urllib.request.urlopen(req, timeout=90) as r:
        total = int(r.headers.get('Content-Length') or 0) + pos
        mode = 'ab' if pos and r.status == 206 else 'wb'
        if mode == 'wb':
            pos = 0
        done = pos
        t0 = time.time()
        last = 0.0
        with open(tmp, mode) as f:
            while True:
                chunk = r.read(262144)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.time()
                if now - last > 0.8:
                    last = now
                    sp = done / max(now - t0, 0.001) / 2**20
                    pct = (done / total * 100) if total else 0
                    sys.stdout.write('\r    %5.1f%%  %7.1f/%7.1f MB  %5.1f MB/s'
                                     % (pct, done / 2**20, total / 2**20, sp))
                    sys.stdout.flush()
    sys.stdout.write('\r    100.0%%  %7.1f MB  完成            \n' % (os.path.getsize(tmp) / 2**20))
    os.replace(tmp, dest)


def main():
    print('=' * 74)
    print('  下载查询编码器 %s' % REPO)
    print('  目标目录：%s' % DEST)
    print('=' * 74)
    ok_base = None
    for base in BASES:
        try:
            urllib.request.urlopen(base, timeout=15).read(1)
            ok_base = base
            break
        except Exception as e:  # noqa: BLE001
            print('  [跳过] 镜像不可达 %s (%s)' % (base, type(e).__name__))
    if not ok_base:
        print('\n[错误] 所有镜像都不可达。请检查网络，或用 U 盘/QQ 传 partB 包。')
        return 1
    print('  使用镜像：%s\n' % ok_base)

    for rel in FILES:
        dest = os.path.join(DEST, rel.replace('/', os.sep))
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print('  [已有] %-34s %8.2f MB' % (rel, os.path.getsize(dest) / 2**20))
            continue
        print('  [下载] %s' % rel)
        last_err = None
        for attempt in range(3):
            try:
                fetch(ok_base, rel, dest)
                last_err = None
                break
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                last_err = e
                print('         第 %d 次失败（%s），重试…' % (attempt + 1, type(e).__name__))
                time.sleep(2)
        if last_err is not None:
            print('\n[错误] %s 下载失败：%s' % (rel, last_err))
            return 1

    print('\n' + '=' * 74)
    print('  完成。现在可以双击 start.bat，或先跑 doctor.bat 体检。')
    print('=' * 74)
    return 0


if __name__ == '__main__':
    sys.exit(main())
