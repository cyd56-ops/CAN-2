"""Phase 5 逐样本诊断工具测试。"""

import numpy as np
import torch

from src.can.v2.crypto.lwe import LWEParams, generate_keypair
from src.can.v2.training.data import CredentialGenerator
from src.can.v2.transformer import (
    ByteTokenizer,
    GatedDecoderTransformer,
    TransformerConfig,
    build_sample_diagnostics,
    generate_synthetic_corpus,
)
from src.can.v2.transformer.plain_model import PlainDecoderTransformer


def _examples():
    """构造单个 public 样本，避免测试触碰 test split。"""
    return generate_synthetic_corpus(7, 1, 1, 1)["train"][:1]


def _config() -> TransformerConfig:
    """返回诊断测试使用的极小 Transformer 配置。"""
    return TransformerConfig(
        max_seq_len=256, num_layers=2, cut_layer=1, d_model=32, num_heads=4, d_ff=64
    )


def test_plain_diagnostics_contains_sample_and_teacher_forced_fields():
    """Plain 诊断应输出逐样本生成和 teacher-forced 字段。"""
    model = PlainDecoderTransformer(_config())
    result = build_sample_diagnostics(
        model, _examples(), ByteTokenizer(), torch.device("cpu"), 1, "kv"
    )
    record = result["records"][0]
    assert result["model_kind"] == "plain"
    assert result["sample_count"] == 1
    assert record["sample_id"].startswith("train-")
    assert "teacher_forced" in record["routes"]["protected"]
    assert isinstance(
        record["routes"]["protected"]["teacher_forced"]["correct_by_position"], list
    )


def test_can_diagnostics_uses_credential_without_serializing_secret():
    """CAN 诊断应走 credential 路由且输出不包含 secret 字段。"""
    params = LWEParams(n=8, m=16)
    matrix, secret, vector = generate_keypair(params, np.random.default_rng(3))
    model = GatedDecoderTransformer(matrix, vector, params, config=_config())
    generator = CredentialGenerator(matrix, secret, vector, params, seed=9)
    result = build_sample_diagnostics(
        model,
        _examples(),
        ByteTokenizer(),
        torch.device("cpu"),
        1,
        "kv",
        generator,
    )
    record_text = repr(result)
    assert result["model_kind"] == "can"
    assert "secret" not in record_text
    assert "protected" in result["records"][0]["routes"]
