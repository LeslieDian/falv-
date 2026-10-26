# -*- coding: utf-8 -*-
"""Qwen3-Embedding 本地封装（= 服务器 scripts/retrieval/embedding_model.py 的 CPU 版副本）。

★ 为什么必须逐位对齐
--------------------
库里的 21.7 万条向量是服务器用 Qwen3-Embedding-0.6B / max_seq_length=2048 /
L2 归一化建出来的。查询向量只要在这三点上有一点不同，就与库向量不在同一
空间，召回会莫名偏低（而且不会报错，只会"感觉不准"）。所以：

    - 模型         : Qwen3-Embedding-0.6B（同一份权重）
    - max_seq_length: 2048
    - normalize     : True
    - 查询侧前缀    : "Instruct: <task>\\nQuery: <q>"   ← 文档侧**不加**

★ 为什么不能用第三方 embedding API 顶替
--------------------------------------
第三方 API 与 Qwen3 的向量空间不同，混用等于把库废掉。要换 embedding
必须**整库重跑**，不是换个 key 就行。
"""
import os

# 与建库端一致的指令描述（查询侧前缀用）
QUERY_INSTRUCTION = (
    "Given a Chinese legal question, case description or issue, "
    "retrieve the most relevant statute provisions and legal documents")

DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_MODEL_SUBDIR = os.path.join("models", "Qwen3-Embedding-0.6B")


class Embedder(object):
    """可用 / 不可用两态；不可用时全站自动降级为「关键词检索」而不是崩掉。

    答辩现场最怕的是环境没配好直接白屏 —— 所以这里一律不抛异常，
    失败信息挂在 .error 上，由前端显式提示"当前为降级模式"。
    """

    def __init__(self, base_dir, model_dir=None, max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
                 batch_size=16, device=None):
        self.base_dir = os.path.abspath(base_dir)
        self.model_dir = model_dir or os.path.join(self.base_dir, DEFAULT_MODEL_SUBDIR)
        self.max_seq_length = int(max_seq_length)
        self.batch_size = int(batch_size)
        self.device = device or os.environ.get("LAW_RAG_DEVICE", "cpu")
        self.instruction = QUERY_INSTRUCTION
        self.model = None
        self.dim = None
        self.error = None
        self._load()

    # ------------------------------------------------------------------ 加载
    def _load(self):
        if not os.path.isdir(self.model_dir):
            self.error = "模型目录不存在：%s" % self.model_dir
            return
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:                                    # noqa: BLE001
            self.error = "sentence-transformers 未安装（%s）" % e
            return
        try:
            m = SentenceTransformer(self.model_dir, device=self.device)
            m.max_seq_length = self.max_seq_length
            dim = None
            for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
                if hasattr(m, attr):
                    dim = int(getattr(m, attr)())
                    break
            if dim is None:
                raise RuntimeError("无法取得模型输出维度")
            self.model = m
            self.dim = dim
        except Exception as e:                                    # noqa: BLE001
            self.error = "模型加载失败：%s" % e
            self.model = None

    @property
    def available(self):
        return self.model is not None

    def info(self):
        if self.available:
            return {"available": True, "model": "Qwen3-Embedding-0.6B",
                    "dim": self.dim, "device": self.device,
                    "max_seq_length": self.max_seq_length, "error": None}
        return {"available": False, "model": None, "dim": None,
                "device": self.device, "max_seq_length": self.max_seq_length,
                "error": self.error}

    # ------------------------------------------------------------------ 编码
    def _encode(self, texts):
        import numpy as np
        v = self.model.encode(texts, batch_size=self.batch_size,
                              convert_to_numpy=True, normalize_embeddings=True,
                              show_progress_bar=False)
        return np.asarray(v, dtype="float32")

    def encode_queries(self, queries):
        """查询侧：加 instruction 前缀。"""
        if isinstance(queries, str):
            queries = [queries]
        pref = self.instruction
        return self._encode(["Instruct: %s\nQuery: %s" % (pref, q) for q in queries])

    def encode_docs(self, texts):
        """文档侧：**不加**前缀（上传新案例入库时用）。"""
        if isinstance(texts, str):
            texts = [texts]
        return self._encode(list(texts))
