# -*- coding: utf-8 -*-
"""一键启动。读 config.json 的 host/port，起 uvicorn。"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)


def main():
    cfg = {"host": "127.0.0.1", "port": 8760}
    p = os.path.join(HERE, "config.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            cfg.update(json.load(f).get("server") or {})
    host, port = cfg["host"], int(cfg["port"])

    # 断网免疫：编码器权重就在本地 models/ 下，不需要联网核对版本。
    # 若不锁死，huggingface_hub 在无网环境会尝试连接并等待超时（启动卡十几秒~失败）。
    if (os.path.isdir(os.path.join(HERE, "models", "Qwen3-Embedding-0.6B"))
            and os.environ.get("LAW_RAG_ALLOW_HF") != "1"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print("=" * 66)
    print("  法律领域智能体 · 离线演示")
    print("  浏览器打开： http://%s:%d" % (host, port))
    print("  按 Ctrl+C 停止")
    print("=" * 66)
    import uvicorn
    uvicorn.run("app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
