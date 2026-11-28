import ctypes
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace as Ns
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

from vllm_ascend.attention.kimi_c8_static import pa_nz_cache_view
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake import base_worker, pull_worker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.base_worker import MooncakeBaseConnectorWorker
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.pull_scheduler import MooncakeSchedulerSendingThread

from .helpers import (
    make_blocks,
    make_full_spec,
    make_kv_cache_tensor,
    make_metadata_groups,
    make_pp_metadata,
    make_request,
    make_transfer_metadata,
)
from .test_pull_scheduler import make_pull_scheduler
from .test_pull_worker import make_thread

PAYLOAD, CONTRACT = 73728, ("kimi_e4m3_pa_nz_native_v2", 0.25)


def spec():
    return AscendMLAAttentionSpec(
        block_size=1536, num_kv_heads=1, head_size=576, dtype=torch.float8_e4m3fn, static_fp8_pa_nz=True
    )


def execute(thread, peer, entries=None, *, remote_tp_rank=0, remote_dcp_size=1):
    return thread._execute_tp_transfer_bucket(
        remote_pp_rank=0,
        remote_pcp_rank=0,
        remote_tp_rank=remote_tp_rank,
        remote_pcp_size=1,
        remote_tp_size=remote_dcp_size,
        remote_dcp_size=remote_dcp_size,
        transfer_entries_by_spec=entries or {0: {(0, 0): [("r", [0], [0])]}},
        remote_metadata=peer,
    )


def map_blocks(thread, remote_block, block_lists, token_counts):
    """Lists: suffix/full/remote; counts: local prompt/remote prompt/computed."""
    thread._kimi_c8_enabled = True
    return thread._compute_group_block_ids(
        "r", [[0]], 1, 0, 1536, remote_block, *block_lists, *token_counts, 24, 24, spec(), 0
    )


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize(
    "world,rank,remote_world,remote_block",
    [(1, 0, 1, 768), (1, 0, 1, 1664), (8, 3, 1, 768), (8, 3, 1, 1664), (8, 7, 8, 1664)],
)
def test_real_transfer_addresses_payload_and_padding(world, rank, group, remote_world, remote_block):
    remote_page, remote_rank = divmod(rank, remote_world)
    local_size, local_stride, remote_stride = 1664, 16 * PAYLOAD, 20 * PAYLOAD
    local = torch.full((20, local_stride), 165, dtype=torch.uint8)
    remote = torch.zeros(20, remote_stride, dtype=torch.uint8)
    local_cache = pa_nz_cache_view(local.view(-1), 20, local_size, page_stride_bytes=local_stride)
    remote_cache = pa_nz_cache_view(remote.view(-1), 20, remote_block, page_stride_bytes=remote_stride)
    dense = ((torch.arange(20 * remote_block).view(20, remote_block, 1) % 127 + torch.arange(576)) % 251).to(
        torch.uint8
    )
    remote_cache.copy_(dense.view(20, remote_block, 18, 32).permute(0, 2, 1, 3).unsqueeze(1))
    local_meta = replace(
        make_transfer_metadata(),
        block_size=local_size,
        layer_block_sizes=[local_size],
        kv_caches_base_addr=[[local.data_ptr()]],
        block_size_scales=[[1]],
        block_strides=[[local_stride]],
        block_lens=[[local_size * 576]],
        kimi_c8_contracts=[CONTRACT],
    )
    peer = replace(
        make_pp_metadata(tp_base_addrs={remote_rank: [[remote.data_ptr()]]}),
        group_indices=[group],
        block_size=remote_block,
        layer_block_sizes=[remote_block],
        block_size_scales=[[1]],
        block_strides=[[remote_stride]],
        block_lens=[[remote_block * 576]],
        kimi_c8_contracts=[CONTRACT],
    )
    thread = make_thread(
        kv_cache_specs=[replace(spec(), block_size=local_size)],
        layer_block_sizes=[local_size],
        local_metadata=local_meta,
        kv_caches_base_addr=[[local.data_ptr()]],
        block_strides=[[local_stride]],
        block_lens=[[local_size * 576]],
        block_size_scales=[[1]],
        dcp_size=world,
        dcp_rank=rank,
    )
    request = Ns(
        local_block_ids=([1],),
        local_full_block_ids=([1],),
        num_computed_tokens=0,
        remote_block_ids=(([19] * 16,) if group else ()) + (list(range(16)),),
        local_num_prompt_tokens=13313,
        remote_num_prompt_tokens=13313,
    )
    buckets, affected = thread._build_transfer_block_buckets(
        peer, [(0, 0)], {(0, 0): [list(range(remote_world))]}, 1, remote_world, {"r": request}, {}
    )
    bucket = (0, remote_rank)
    assert affected == {bucket: {"r"}}
    _, local_ids, remote_ids = buckets[bucket][0][(0, 0)][0]
    assert local_ids == list(range(13, 26))
    assert remote_ids == list(range(remote_page * 13, (remote_page + 1) * 13))

    def read(_session, destinations, sources, lengths):
        for dst, src, size in zip(destinations, sources, lengths):
            ctypes.memmove(dst, src, size)
        return 0

    thread.engine = MagicMock()
    thread.engine.batch_transfer_sync_read.side_effect = read
    execute(thread, peer, buckets[bucket], remote_tp_rank=remote_rank, remote_dcp_size=remote_world)
    indices = torch.cat([torch.arange(index * 128, (index + 1) * 128) for index in remote_ids])
    expected = dense.reshape(-1, 576)[indices]
    actual = local_cache[1, 0].permute(1, 0, 2).reshape(local_size, 576)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    args = thread.engine.batch_transfer_sync_read.call_args.args
    assert sum(args[3]) == local_size * 576
    if remote_block == local_size:
        assert args[1:] == (
            [local.data_ptr() + local_stride],
            [remote.data_ptr() + remote_page * remote_stride],
            [local_size * 576],
        )
    assert local[0].eq(165).all() and local[2:].eq(165).all()
    assert local[1, local_size * 576 :].eq(165).all()


@pytest.mark.parametrize("padding", [(0, 0), (PAYLOAD, 0), (0, PAYLOAD)])
def test_c8_reuses_request_spans_and_coalesces_only_dense_pages(padding):
    page = 1664
    payload = page * 576
    local_stride, remote_stride = (payload + amount for amount in padding)
    layer_names = ["mla.0", "mla.1"]
    local_bases, remote_bases = [[10_000_000], [20_000_000]], [[30_000_000], [40_000_000]]
    thread = make_thread(
        kv_cache_specs=[replace(spec(), block_size=page)],
        layer_names=layer_names,
        layer_block_sizes=[page] * 2,
        local_metadata=replace(make_transfer_metadata(layer_names=layer_names), kimi_c8_contracts=[CONTRACT] * 2),
        kv_caches_base_addr=local_bases,
        block_strides=[[local_stride]] * 2,
        block_lens=[[payload]] * 2,
    )
    peer = replace(
        make_pp_metadata(
            layer_names=layer_names,
            layer_block_sizes=[page] * 2,
            block_strides=[[remote_stride]] * 2,
            block_lens=[[payload]] * 2,
            tp_base_addrs={0: remote_bases},
        ),
        kimi_c8_contracts=[CONTRACT] * 2,
    )
    thread.engine = MagicMock()
    thread.engine.batch_transfer_sync_read.return_value = 0
    # Two complete pages followed by one 128-token slice of the next page.
    entries = [("r", list(range(27)), list(range(13, 40)))]
    with patch.object(
        pull_worker, "group_concurrent_contiguous", wraps=pull_worker.group_concurrent_contiguous
    ) as group:
        execute(thread, peer, {0: {(0, 0): entries, (1, 1): entries}})

    _, local, remote, sizes = thread.engine.batch_transfer_sync_read.call_args.args
    for (local_base,), (remote_base,) in zip(local_bases, remote_bases):
        full = (
            [(local_base, remote_base + remote_stride, 2 * payload)]
            if not any(padding)
            else [
                (local_base, remote_base + remote_stride, payload),
                (local_base + local_stride, remote_base + 2 * remote_stride, payload),
            ]
        )
        tail = [
            (
                local_base + 2 * local_stride + tile * page * 32,
                remote_base + 3 * remote_stride + tile * page * 32,
                128 * 32,
            )
            for tile in range(18)
        ]
        count = len(full) + len(tail)
        assert list(zip(local[:count], remote[:count], sizes[:count])) == full + tail
        local, remote, sizes = local[count:], remote[count:], sizes[count:]
    assert not sizes and group.call_count == 1


@pytest.mark.parametrize(
    "c8,contract",
    [(True, None), (True, (CONTRACT[0], 0.5)), (True, ("other", 0.25)), (True, "absent"), (False, "absent")],
)
def test_peer_contract_checked_before_any_transfer(c8, contract):
    peer = make_pp_metadata()
    if contract == "absent":
        peer = Ns(**{key: value for key, value in vars(peer).items() if key != "kimi_c8_contracts"})
    else:
        peer = replace(peer, kimi_c8_contracts=[contract])
    local = replace(make_transfer_metadata(), kimi_c8_contracts=[CONTRACT] if c8 else [])
    thread = make_thread(kv_cache_specs=[spec() if c8 else make_full_spec()], local_metadata=local)
    thread.engine = MagicMock()
    thread.engine.batch_transfer_sync_read.return_value = 0
    if c8:
        with pytest.raises(ValueError, match="format/static KV descale mismatch"):
            execute(thread, peer)
        thread.engine.batch_transfer_sync_read.assert_not_called()
    else:
        execute(thread, peer)
        thread.engine.batch_transfer_sync_read.assert_called_once()
    with pytest.raises(ValueError, match="one contract per layer"):
        replace(make_transfer_metadata(layer_names=["mla.0", "mla.1"]), kimi_c8_contracts=[CONTRACT])


def test_registration_reads_impl_scale_and_preserves_physical_stride(monkeypatch):
    backing = torch.zeros(2 * 16 * PAYLOAD, dtype=torch.uint8)
    cache = pa_nz_cache_view(backing, 2, 1536, page_stride_bytes=16 * PAYLOAD)
    worker = MooncakeBaseConnectorWorker.__new__(MooncakeBaseConnectorWorker)
    worker.vllm_config = Ns(
        compilation_config=Ns(
            static_forward_context={
                "mla": Ns(fak_descale_float=torch.tensor([0.5]), impl=Ns(fak_descale_float=torch.tensor([0.25])))
            }
        )
    )
    worker.ascend_config = Ns(kvpp_config=Ns(size=1))
    worker.pcp_rank, worker.tp_rank, worker.tp_size = 0, 0, 1
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            make_kv_cache_tensor(
                size=backing.nbytes, layers=["mla"], layer_stride=backing.nbytes, block_stride=16 * PAYLOAD
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["mla"], kv_cache_spec=spec())],
    )
    worker.engine_id, worker.te_rpc_port = "p", 9000
    worker.block_size, worker.side_channel_host, worker.handshake_port = 1536, "127.0.0.1", 5000
    monkeypatch.setattr(base_worker, "global_te", MagicMock())
    worker.register_kv_caches({"mla": (cache,)})
    metadata = worker.xfer_handshake_metadata
    sender = MooncakeSchedulerSendingThread.__new__(MooncakeSchedulerSendingThread)
    sender.tp_size = sender.pp_size = sender.pcp_size = 1
    sender.engine_id, sender.use_kv_pp = "p", False
    merged = sender._merge_metadata_by_pp_rank({0: metadata})[0]
    assert merged.kimi_c8_contracts == [CONTRACT] and merged.block_size_scales == [[1]]
    assert merged.block_lens == [[1536 * 576]]


@pytest.mark.parametrize("failed", [False, True])
def test_ready_waits_for_all_c8_mamba_draft_tasks_even_after_failure(failed):
    thread = make_thread()
    peer = make_metadata_groups(
        tp_size=3, pp_metadata=make_pp_metadata(tp_base_addrs={rank: [[1000]] for rank in range(3)})
    )
    thread._get_remote_metadata = MagicMock(return_value=peer)
    thread.remote_tp_rank_groups = {"engine-p": {0: {(0, 0): [[0], [1], [2]]}}}
    thread.remote_layer_index_pairs = {"engine-p": {0: [(0, 0)]}}
    buckets = {(0, rank): {rank: {(0, 0): [("r", [1], [2])]}} for rank in range(3)}
    thread._build_transfer_block_buckets = MagicMock(return_value=(buckets, {(0, rank): {"r"} for rank in range(3)}))
    futures, submitted = [Future() for _ in range(3)], threading.Event()
    futures[0].set_exception(ValueError("C8 failure")) if failed else futures[0].set_result(None)
    futures[1].set_result(None)

    def submit(_function, _pp, _pcp, tp, *args):
        if tp == 2:
            submitted.set()
        return futures[tp]

    thread.executor = Ns(submit=submit)
    with ThreadPoolExecutor(max_workers=1) as runner:
        job = runner.submit(thread._handle_requests, "engine-p", "host", 6000, {"r": Ns()})
        try:
            assert submitted.wait(timeout=2) and not job.done()
        finally:
            futures[2].set_result(None)
        assert job.result(timeout=2) == ({"r"} if failed else set())


@pytest.mark.parametrize("remote_block", [768, 1536])
def test_scheduler_to_worker_retains_dcp_prefix_offset(remote_block):
    scheduler = make_pull_scheduler()
    scheduler.need_truncate, scheduler.dcp_size = True, 8
    scheduler.group_block_size, scheduler.group_unique_specs = [1536], [[spec()]]
    request = make_request(
        prompt_token_ids=list(range(24577)),
        num_prompt_tokens=24577,
        kv_transfer_params=dict(
            do_remote_prefill=True,
            remote_block_ids=(list(range(40, 80)),),
            remote_engine_id="engine-p",
            remote_host="host",
            remote_port=6000,
            remote_request_id="p",
            remote_num_prompt_tokens=24576,
        ),
    )
    assert "num_computed_tokens" not in request.kv_transfer_params
    count, asynchronous = scheduler.get_num_new_matched_tokens(request, 12288)
    assert asynchronous and count == 12288
    scheduler.update_state_after_alloc(request, make_blocks(unhashed=([9],), full=([7, 9],)), count)
    metadata = scheduler.build_connector_meta(MagicMock()).requests[request.request_id]
    assert metadata.num_computed_tokens == 12288
    mapped = map_blocks(
        make_thread(dcp_size=8, dcp_rank=3),
        remote_block,
        (metadata.local_block_ids[0], metadata.local_full_block_ids[0], metadata.remote_block_ids[0]),
        (metadata.local_num_prompt_tokens, metadata.remote_num_prompt_tokens, metadata.num_computed_tokens),
    )
    first = (40 + (12288 + 3 * 1536) // remote_block) * (remote_block // 128)
    assert mapped == [(0, list(range(108, 120)), list(range(first, first + 12)))]


@pytest.mark.parametrize("length", [128, 129, 130, 1536, 1537, 1538])
@pytest.mark.parametrize("rank", [0, 1, 7])
@pytest.mark.parametrize("truncated", [False, True])
def test_unequal_pages_transfer_exactly_the_history_before_tail_recompute(length, rank, truncated):
    mapped = map_blocks(
        make_thread(dcp_size=8, dcp_rank=rank),
        768,
        ([9], [9], list(range(40, 44))),
        (length, length - int(truncated), 0),
    )
    history_end = length - int(truncated)
    offsets = [i for i in range(12) if rank * 1536 + i * 128 < history_end]
    assert mapped == ([(0, [108 + i for i in offsets], [240 + rank * 12 + i for i in offsets])] if offsets else [])


@pytest.mark.parametrize("remote_dcp", [2, 4])
@pytest.mark.parametrize("prompt_len", [500, 1664, 1665, 3328, 3329, 13313, 227287])
def test_c8_sub_tp_dcp_producer_to_dcp8_consumer_maps_exact_prompt_pages(remote_dcp, prompt_len):
    page = 1664
    local_dcp = 8
    c8_spec = replace(spec(), block_size=page)
    remote_blocks = list(range(100, 100 + (prompt_len + page * remote_dcp - 1) // (page * remote_dcp)))
    local_blocks = list(range(500, 500 + (prompt_len + page * local_dcp - 1) // (page * local_dcp)))
    expected_pages = (prompt_len + page - 1) // page

    transferred: dict[int, tuple[int, int]] = {}
    for d_rank in range(local_dcp):
        thread = make_thread(dcp_size=local_dcp, dcp_rank=d_rank, tp_size=8, tp_rank=d_rank)
        thread._kimi_c8_enabled = True
        mapped = thread._compute_group_block_ids(
            request_id="req",
            remote_tp_rank_groups=[list(range(8))],
            remote_dcp_size=remote_dcp,
            spec_index=0,
            local_block_size=page,
            remote_block_size=page,
            local_group_block_ids=local_blocks,
            local_full_group_block_ids=local_blocks,
            remote_group_block_ids=remote_blocks,
            local_num_prompt_tokens=prompt_len,
            remote_num_prompt_tokens=prompt_len,
            num_computed_tokens=0,
            local_block_size_scale=1,
            remote_block_size_scale=1,
            spec=c8_spec,
            selection_index=d_rank,
        )
        for remote_tp_rank, local_kblocks, remote_kblocks in mapped:
            assert len(local_kblocks) == len(remote_kblocks)
            assert len(local_kblocks) % 13 == 0
            for idx in range(0, len(local_kblocks), 13):
                local_page_id = local_kblocks[idx] // 13
                remote_page_id = remote_kblocks[idx] // 13
                assert local_kblocks[idx : idx + 13] == list(range(local_page_id * 13, (local_page_id + 1) * 13))
                assert remote_kblocks[idx : idx + 13] == list(range(remote_page_id * 13, (remote_page_id + 1) * 13))
                local_pos = local_blocks.index(local_page_id)
                global_page = local_pos * local_dcp + d_rank
                assert global_page not in transferred
                transferred[global_page] = (remote_tp_rank % remote_dcp, remote_blocks.index(remote_page_id))

    assert sorted(transferred) == list(range(expected_pages))
    for global_page, (src_dcp_rank, src_block_idx) in transferred.items():
        assert src_dcp_rank == global_page % remote_dcp
        assert src_block_idx == global_page // remote_dcp
