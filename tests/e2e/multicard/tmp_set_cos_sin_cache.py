import torch

import torch.nn.functional as F

import pytest

from unittest.mock import MagicMock

from vllm_ascend.ops.rotary_embedding import __set_cos_sin_cache


class RotaryEmbedding(torch.nn.Module):

    def __init__(self, base, rotary_dim, max_position_embeddings):
        super().__init__()

        self.base = base

        self.rotary_dim = rotary_dim

        self.max_position_embeddings = max_position_embeddings

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        return __set_cos_sin_cache(self, seq_len, device, dtype)


def test_set_cos_sin_cache_registers_buffers_and_sets_embed():
    # prepare an instance with reasonable values

    base = 10000.0

    rotary_dim = 4

    max_pos = 10

    model = RotaryEmbedding(base, rotary_dim, max_pos)

    # mock out register_buffer

    model.register_buffer = MagicMock()

    # call the private method via name mangling

    model._RotaryEmbedding._set_cos_sin_cache(seq_len=8, device="cpu", dtype=torch.float32)

    # expect three calls: inv_freq, cos, sin

    assert model.register_buffer.call_count == 3

    names = [call.args[0] for call in model.register_buffer.call_args_list]

    assert set(names) == {"inv_freq", "cos", "sin"}

    # verify inv_freq shape

    inv_freq = model.register_buffer.call_args_list[0].args[1]

    assert isinstance(inv_freq, torch.Tensor)

    assert inv_freq.shape == (rotary_dim // 2,)

    # verify cos buffer

    cos = model.register_buffer.call_args_list[1].args[1]

    assert isinstance(cos, torch.Tensor)

    assert cos.shape == (max_pos, rotary_dim)

    assert cos.dtype == torch.float32

    # verify sin buffer

    sin = model.register_buffer.call_args_list[2].args[1]

    assert isinstance(sin, torch.Tensor)

    assert sin.shape == (max_pos, rotary_dim)

    assert sin.dtype == torch.float32

    # ensure embed attribute is set correctly

    assert model.embed is F.embedding
