from jaxchat.presets import PRESETS
from jaxchat.model import expected_parameter_breakdown
from jaxchat.tools import execute_python


def test_0p56b_rust65k_preset_shape():
    cfg = PRESETS["0p56b-rust65k"]
    bd = expected_parameter_breakdown(cfg)
    assert cfg.vocab_size == 65_536
    assert cfg.depth == 20
    assert cfg.d_model == 1280
    assert cfg.n_heads == 10
    assert cfg.n_kv_heads == 10
    assert cfg.n_value_layers == 0
    assert cfg.bigram_hash_embed is False
    assert cfg.n_train_iters == 21_400
    assert cfg.actual_train_tokens == 11_219_763_200
    assert bd["total"] - bd["scalars"] == 560_988_160


def test_python_tool_executes_expression():
    result = execute_python("37 * 42", timeout_s=1.0)
    assert result["ok"]
    assert result["output"] == "1554"


def test_python_tool_times_out():
    result = execute_python("while True:\n    pass", timeout_s=0.1)
    assert not result["ok"]
    assert "TimeoutExpired" in result["output"]
