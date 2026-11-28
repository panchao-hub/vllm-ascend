from types import SimpleNamespace as Ns
from unittest.mock import Mock

import pytest
import torch
from vllm.config.compilation import CUDAGraphMode as Mode

from vllm_ascend.ascend_config import AscendConfig


@pytest.fixture
def configuration():
    config = Ns(
        model_config=Ns(
            hf_config=Ns(model_type="kimi_linear", kv_lora_rank=512, qk_rope_head_dim=64, mla_use_output_gate=False)
        ),
        mla_cache_mode="C8",
        compilation_config=Ns(cudagraph_mode=Mode.NONE),
        parallel_config=Ns(
            prefill_context_parallel_size=1, decode_context_parallel_size=8, cp_kv_cache_interleave_size=1664
        ),
        kv_transfer_config=Ns(is_kv_producer=False, kv_connector="MooncakePullConnector"),
        cache_config=Ns(block_size=1664, enable_prefix_caching=True),
        use_v2_model_runner=True,
        speculative_config=None,
    )
    config.model_config.hf_text_config = config.model_config.hf_config
    return config


@pytest.mark.parametrize("value,expected", [("default", "C16"), ("c8", "C8"), (" C16 ", "C16")])
def test_configuration_mode_normalization_and_c16_passthrough(value, expected):
    kwargs = {} if value == "default" else {"mla_cache_mode": value}
    options = AscendConfig(sparse_kv_offload_config=Ns(enabled=False), **kwargs)
    assert options.mla_cache_mode == expected
    if expected == "C16":
        options._validate_mla_cache_mode(None)  # Existing models must not be inspected.


@pytest.mark.parametrize("value", ["INT8", "FP8", "", None, True, 8])
def test_configuration_rejects_invalid_mode(value):
    with pytest.raises(ValueError, match="mla_cache_mode"):
        AscendConfig(mla_cache_mode=value, sparse_kv_offload_config=Ns(enabled=False))


@pytest.mark.parametrize(
    "owner,field,value,error",
    [
        (None, None, None, None),
        ("root", "speculative_config", Ns(method="dspark"), None),
        ("model", "model_type", "deepseek_v3", "requires Kimi K3"),
        ("model", "kv_lora_rank", 256, "requires Kimi K3"),
        ("model", "qk_rope_head_dim", 0, "requires Kimi K3"),
        ("root", "use_v2_model_runner", False, "RunnerV2"),
        ("root", "speculative_config", Ns(method="eagle"), "DSpark"),
        ("parallel", "prefill_context_parallel_size", 2, "prefill_context_parallel"),
        ("parallel", "decode_context_parallel_size", 2, None),
        ("parallel", "decode_context_parallel_size", 4, None),
        ("parallel", "decode_context_parallel_size", 3, "decode_context_parallel"),
        ("parallel", "decode_context_parallel_size", 16, "decode_context_parallel"),
        ("parallel", "cp_kv_cache_interleave_size", 128, "interleave"),
        ("transfer", "is_kv_producer", True, None),
        ("transfer", "kv_connector", "MooncakeConnectorV1", "supported Mooncake"),
        ("cache", "block_size", 128, "block_size=1664"),
    ],
)
def test_configuration_checks_model_runner_pd_and_parallel_boundaries(configuration, owner, field, value, error):
    if owner:
        target = {
            "root": configuration,
            "model": configuration.model_config.hf_text_config,
            "parallel": configuration.parallel_config,
            "transfer": configuration.kv_transfer_config,
            "cache": configuration.cache_config,
        }[owner]
        setattr(target, field, value)
    options = AscendConfig(mla_cache_mode="C8", sparse_kv_offload_config=Ns(enabled=False))
    if error:
        with pytest.raises(ValueError, match=error):
            options._validate_mla_cache_mode(configuration)
    else:
        before = vars(configuration.model_config.hf_text_config).copy()
        options._validate_mla_cache_mode(configuration)
        assert vars(configuration.model_config.hf_text_config) == before


def test_c8_dcp8_producer_allows_store_with_native_pull(configuration):
    transfer = configuration.kv_transfer_config
    transfer.is_kv_producer = True
    transfer.kv_connector = "MultiConnector"
    pull = {"kv_connector": "MooncakeConnectorV2", "kv_role": "kv_producer"}
    transfer.kv_connector_extra_config = {
        "connectors": [pull, {"kv_connector": "AscendStoreConnector", "kv_role": "kv_producer"}]
    }
    options = AscendConfig(mla_cache_mode="C8", sparse_kv_offload_config=Ns(enabled=False))
    options._validate_mla_cache_mode(configuration)

    configuration.cache_config.enable_prefix_caching = False
    with pytest.raises(ValueError, match="enable_prefix_caching"):
        options._validate_mla_cache_mode(configuration)
    configuration.cache_config.enable_prefix_caching = True

    pull["kv_connector"] = "MooncakeHybridConnector"
    with pytest.raises(ValueError, match="requires a supported Mooncake connector"):
        options._validate_mla_cache_mode(configuration)


def test_c8_store_reports_async_option_ignored_at_multi_connector_level(configuration, monkeypatch):
    configuration.kv_transfer_config = Ns(
        kv_connector="MultiConnector",
        is_kv_producer=True,
        kv_connector_extra_config={
            "load_async": True,
            "connectors": [{"kv_connector": "MooncakeConnectorV2"}, {"kv_connector": "AscendStoreConnector"}],
        },
    )
    options = AscendConfig(mla_cache_mode="C8", sparse_kv_offload_config=Ns(enabled=False))
    warning = Mock()
    monkeypatch.setattr("vllm_ascend.ascend_config.logger.warning", warning)

    options._validate_mla_cache_mode(configuration)

    warning.assert_called_once()
    assert "AscendStoreConnector" in warning.call_args.args[0]
    assert warning.call_args.args[1:] == ("load_async", True, False)


@pytest.mark.parametrize("dcp", [2, 4, 8])
def test_c8_dcp_enforces_interleave_block_size_and_store_prefix_caching(configuration, dcp):
    configuration.parallel_config.decode_context_parallel_size = dcp
    transfer = configuration.kv_transfer_config
    transfer.is_kv_producer = True
    transfer.kv_connector = "MultiConnector"
    transfer.kv_connector_extra_config = {
        "connectors": [
            {"kv_connector": "MooncakeConnectorV2", "kv_role": "kv_producer"},
            {"kv_connector": "AscendStoreConnector", "kv_role": "kv_producer"},
        ]
    }
    options = AscendConfig(mla_cache_mode="C8", sparse_kv_offload_config=Ns(enabled=False))
    options._validate_mla_cache_mode(configuration)

    configuration.parallel_config.cp_kv_cache_interleave_size = 128
    with pytest.raises(ValueError, match="interleave"):
        options._validate_mla_cache_mode(configuration)
    configuration.parallel_config.cp_kv_cache_interleave_size = 1664

    configuration.cache_config.block_size = 128
    with pytest.raises(ValueError, match="block_size=1664"):
        options._validate_mla_cache_mode(configuration)
    configuration.cache_config.block_size = 1664

    configuration.cache_config.enable_prefix_caching = False
    with pytest.raises(ValueError, match="enable_prefix_caching"):
        options._validate_mla_cache_mode(configuration)


@pytest.mark.parametrize("padded_stride", [False, True])
@pytest.mark.parametrize("with_invalid_slots", [False, True])
def test_c8_cache_write_plan_matches_write_static_fp8_cache(padded_stride, with_invalid_slots):
    from vllm_ascend.attention.kimi_c8_static import C8CacheWritePlan, pa_nz_cache_view, write_static_fp8_cache

    torch.manual_seed(7)
    num_blocks, block_size = 4, 128
    page_stride = (block_size + 128) * 576 if padded_stride else block_size * 576
    backing_ref = torch.zeros(num_blocks * page_stride, dtype=torch.uint8)
    backing_opt = torch.zeros(num_blocks * page_stride, dtype=torch.uint8)
    cache_ref = pa_nz_cache_view(backing_ref, num_blocks, block_size, page_stride_bytes=page_stride)
    cache_opt = pa_nz_cache_view(backing_opt, num_blocks, block_size, page_stride_bytes=page_stride)

    merged = torch.randn(300, 576, dtype=torch.bfloat16)
    slots = torch.arange(300, dtype=torch.int64)
    if with_invalid_slots:
        slots[::17] = -1
    scale = torch.tensor([1.25], dtype=torch.float32)

    write_static_fp8_cache(merged, cache_ref, slots, scale)
    plan = C8CacheWritePlan.prepare(cache_opt, slots)
    assert plan.all_valid is (not with_invalid_slots)
    plan.write(merged, cache_opt, slots, scale)

    assert torch.equal(backing_ref, backing_opt)

    bad_slots = torch.tensor([num_blocks * block_size], dtype=torch.int64)
    with pytest.raises(ValueError, match="capacity"):
        C8CacheWritePlan.prepare(cache_opt, bad_slots)


@pytest.mark.parametrize("padded_stride", [False, True])
def test_c8_prefill_history_block_fast_path_and_fallbacks_are_bit_exact(padded_stride):
    from vllm_ascend.attention.kimi_c8_static import (
        C8PrefillHistory,
        _dequantize_static_fp8,
        _gather_static_fp8_rows,
        pa_nz_cache_view,
        write_static_fp8_cache,
    )

    torch.manual_seed(11)
    num_blocks, block_size = 8, 128
    page_stride = (block_size + 128) * 576 if padded_stride else block_size * 576
    backing = torch.zeros(num_blocks * page_stride, dtype=torch.uint8)
    cache = pa_nz_cache_view(backing, num_blocks, block_size, page_stride_bytes=page_stride)
    merged = torch.randn(num_blocks * block_size, 576, dtype=torch.bfloat16)
    write_static_fp8_cache(merged, cache, torch.arange(num_blocks * block_size, dtype=torch.int64), 1.0)

    table = torch.tensor([[5, 2, 7, 0], [3, 1, 6, 4]], dtype=torch.int32)
    # Chunk 0: both reqs aligned full blocks (fast path)
    # Chunk 1: req 0 full block + req 1 partial tail (fast path)
    # Chunk 2: unaligned start on req 0 (fallback path)
    # Chunk 3: interior partial block on req 0 + full block on req 1 (fallback path)
    lengths = torch.tensor([[256, 128], [128, 73], [90, 0], [50, 128]], dtype=torch.int32)
    starts = torch.tensor([[0, 0], [256, 128], [37, 0], [0, 128]], dtype=torch.int32)
    scale = torch.tensor([1.0], dtype=torch.float32)

    history = C8PrefillHistory.prepare(cache, table, lengths, starts)
    assert history.chunk_block_ids[0] is not None
    assert history.chunk_block_ids[1] is not None
    assert history.chunk_block_ids[2] is None
    assert history.chunk_block_ids[3] is None

    for chunk_idx in range(lengths.shape[0]):
        actual = history.read(cache, chunk_idx, scale, dtype=torch.bfloat16)
        ref_rows = _gather_static_fp8_rows(
            cache, history.block_table, history.lengths[chunk_idx], history.starts[chunk_idx], history.totals[chunk_idx]
        )
        expected = _dequantize_static_fp8(ref_rows, scale, torch.bfloat16).unsqueeze(1)
        assert torch.equal(actual, expected)


def test_c8_prefill_history_read_raw_and_dequant_contract():
    from vllm_ascend.attention.kimi_c8_static import (
        C8PrefillHistory,
        _dequantize_static_fp8,
        pa_nz_cache_view,
        supports_npu_fp8_anti_quant,
        write_static_fp8_cache,
    )

    torch.manual_seed(17)
    num_blocks, block_size = 4, 128
    backing = torch.zeros(num_blocks * block_size * 576, dtype=torch.uint8)
    cache = pa_nz_cache_view(backing, num_blocks, block_size)
    merged = torch.randn(num_blocks * block_size, 576, dtype=torch.bfloat16)
    scale = torch.tensor([0.1372941], dtype=torch.float32)
    write_static_fp8_cache(merged, cache, torch.arange(num_blocks * block_size, dtype=torch.int64), scale)

    table = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32)
    lengths = torch.tensor([[128, 64], [0, 0]], dtype=torch.int32)
    starts = torch.tensor([[0, 0], [0, 0]], dtype=torch.int32)
    history = C8PrefillHistory.prepare(cache, table, lengths, starts)

    raw0 = history.read_raw(cache, 0)
    assert raw0.dtype == torch.uint8
    assert raw0.shape == (192, 1, 576)
    assert torch.equal(
        _dequantize_static_fp8(raw0, scale, torch.bfloat16), history.read(cache, 0, scale, dtype=torch.bfloat16)
    )

    raw_empty = history.read_raw(cache, 1)
    assert raw_empty.dtype == torch.uint8
    assert raw_empty.shape == (0, 1, 576)
    deq_empty = _dequantize_static_fp8(raw_empty, scale, torch.bfloat16)
    assert deq_empty.dtype == torch.bfloat16
    assert deq_empty.shape == (0, 1, 576)

    assert not supports_npu_fp8_anti_quant(torch.device("cpu"), torch.bfloat16)
    assert not supports_npu_fp8_anti_quant(torch.device("cpu"), torch.float32)


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("all_valid", [True, False])
def test_c8_quantize_and_write_plan_preserve_caller_input(input_dtype, all_valid):
    from vllm_ascend.attention.kimi_c8_static import (
        C8CacheWritePlan,
        pa_nz_cache_view,
        quantize_static_fp8,
    )

    sample = torch.tensor([4.0, -8.0, 1000.0, -1000.0], dtype=input_dtype)
    sample_before = sample.clone()
    _ = quantize_static_fp8(sample, 2.0)
    assert torch.equal(sample, sample_before)

    num_blocks, block_size = 2, 128
    backing = torch.zeros(num_blocks * block_size * 576, dtype=torch.uint8)
    cache = pa_nz_cache_view(backing, num_blocks, block_size)
    merged = torch.randn(16, 576, dtype=input_dtype) * 10.0
    merged_before = merged.clone()
    slots = torch.arange(16, dtype=torch.int64)
    if not all_valid:
        slots[0] = -1
    plan = C8CacheWritePlan.prepare(cache, slots)
    assert plan.all_valid is all_valid
    plan.write(merged, cache, slots, torch.tensor([2.0], dtype=torch.float32))
    assert torch.equal(merged, merged_before)


def test_c8_anti_quant_probe_checks_real_layout_and_parity():
    from vllm_ascend.attention.kimi_c8_static import (
        K3_MLA_CACHE_WIDTH,
        _probe_npu_fp8_anti_quant,
        resolve_npu_fp8_anti_quant,
        supports_npu_fp8_anti_quant,
    )

    assert resolve_npu_fp8_anti_quant(torch.device("cpu"), torch.bfloat16) is None
    assert supports_npu_fp8_anti_quant(torch.device("npu"), torch.float32) is False

    seen: dict[str, object] = {}

    def exact_ref_anti_quant(x, s, *, dst_dtype):
        seen["shape"] = tuple(x.shape)
        seen["dtype"] = x.dtype
        seen["nonzero"] = int((x.view(torch.uint8) != 0).sum().item())
        return x.float().mul(s).to(dst_dtype)

    assert _probe_npu_fp8_anti_quant(exact_ref_anti_quant, torch.device("cpu"), torch.bfloat16) is True
    assert seen["shape"] == (2, 1, K3_MLA_CACHE_WIDTH)
    assert seen["dtype"] == torch.float8_e4m3fn
    assert seen["nonzero"] == 2 * K3_MLA_CACHE_WIDTH

    def wrong_anti_quant(x, s, *, dst_dtype):
        return torch.zeros(x.shape, dtype=dst_dtype, device=x.device)

    assert _probe_npu_fp8_anti_quant(wrong_anti_quant, torch.device("cpu"), torch.bfloat16) is False

    def unsupported_anti_quant(x, s, *, dst_dtype):
        raise RuntimeError("AclNN_Parameter_Error(EZ1001): dtype not support")

    assert _probe_npu_fp8_anti_quant(unsupported_anti_quant, torch.device("cpu"), torch.bfloat16) is False


def test_c8_project_prefill_kv_matches_unquantized_kv_b_proj():
    from vllm_ascend.attention.mla_v1 import AscendMLAImpl

    torch.manual_seed(7)
    num_heads, kv_lora_rank, qk_nope_dim, v_dim = 4, 16, 12, 8
    weight_out_in = torch.randn(num_heads * (qk_nope_dim + v_dim), kv_lora_rank, dtype=torch.float32)
    kv_c_normed = torch.randn(5, kv_lora_rank, dtype=torch.float32)

    kv_b_weight = weight_out_in.t().view(kv_lora_rank, num_heads, qk_nope_dim + v_dim)
    w_uk, w_uv = kv_b_weight.split([qk_nope_dim, v_dim], dim=-1)

    impl = AscendMLAImpl.__new__(AscendMLAImpl)
    impl.num_heads = num_heads
    impl.kv_lora_rank = kv_lora_rank
    impl.qk_nope_head_dim = qk_nope_dim
    impl.v_head_dim = v_dim
    impl.W_UK_T = w_uk.permute(1, 2, 0).contiguous()
    impl._c8_w_uv_prefill = w_uv.reshape(kv_lora_rank, num_heads * v_dim).contiguous()

    k_nope, v = impl._project_c8_prefill_kv(kv_c_normed)
    ref = torch.nn.functional.linear(kv_c_normed, weight_out_in).view(-1, num_heads, qk_nope_dim + v_dim)
    ref_k, ref_v = ref.split([qk_nope_dim, v_dim], dim=-1)
    torch.testing.assert_close(k_nope, ref_k)
    torch.testing.assert_close(v, ref_v)
