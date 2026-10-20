import pickle
import sys
import time
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm import kernels
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler, verify_drafts
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

# torch.compile (RMSNorm, rotary, SiluAndMul, Sampler) keeps at most recompile_limit compiled versions of a function
# (one per tensor rank, hidden size, static/dynamic shape, default device...). Past the limit it silently runs the
# function uncompiled, which rounds differently: Qwen3-8B with a 0.6B draft model went past the default 8 for
# rms_forward while recording CUDA graphs, so graphs and eager gave different tokens (P5b2). Allow plenty, and fail
# loudly instead of changing numbers if that is ever not enough.
torch._dynamo.config.recompile_limit = 64
torch._dynamo.config.fail_on_recompile_limit_hit = True


def batch_buckets(max_bs: int) -> list[int]:
    """Batch sizes CUDA graphs are recorded for: 1, 2, 4, 8, then every 16, then max_bs itself."""
    return [x for x in [1, 2, 4, 8] if x < max_bs] + list(range(16, max_bs, 16)) + [max_bs]


def mixed_batch(seqs: list[Sequence], k: int, block_size: int) -> dict:
    """The flat batch of a mixed step, as Python lists (see ModelRunner.prepare_mixed). Each request brings its
    own query tokens: a decode request its last token plus k guess slots (padded like prepare_decode), a prefill
    request its chunk. Attention runs as varlen over the paged cache for all of them: this step's KV is stored
    first, then every query reads positions 0 .. its own position (causal, aligned to the bottom right), so
    cu_seqlens_k counts each request's whole length up to the end of its query tokens. logits_indices picks the
    rows that need logits: all 1+k of a decode request, the last row of a prefill request's last chunk, none for
    an earlier chunk (it produces no token)."""
    input_ids, positions, slot_mapping, logits_indices = [], [], [], []
    cu_seqlens_q, cu_seqlens_k = [0], [0]
    for seq in seqs:
        if seq.is_prefill:
            start, end = seq.num_cached_tokens, seq.num_cached_tokens + seq.num_scheduled_tokens
            input_ids.extend(seq[start:end])
        else:
            start, end = len(seq) - 1, len(seq) + k
            tokens = [seq.last_token] + seq.draft_token_ids
            input_ids.extend(tokens + [tokens[-1]] * (1 + k - len(tokens)))
        positions.extend(range(start, end))
        for i in range(start // block_size, (end + block_size - 1) // block_size):
            block_start = seq.block_table[i] * block_size
            slot_mapping.extend(range(block_start + max(start - i * block_size, 0), block_start + min(end - i * block_size, block_size)))
        q0 = cu_seqlens_q[-1]
        cu_seqlens_q.append(q0 + end - start)
        cu_seqlens_k.append(cu_seqlens_k[-1] + end)
        if not seq.is_prefill:
            logits_indices.extend(range(q0, q0 + 1 + k))
        elif end == seq.num_tokens:
            logits_indices.append(q0 + end - start - 1)
    return dict(input_ids=input_ids, positions=positions, slot_mapping=slot_mapping, logits_indices=logits_indices,
                cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max(b - a for a, b in zip(cu_seqlens_q, cu_seqlens_q[1:])),
                max_seqlen_k=max(b - a for a, b in zip(cu_seqlens_k, cu_seqlens_k[1:])))


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        kernels.backend = config.kernel_backend    # read by RMSNorm, SiluAndMul and Qwen3Attention when they are built
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        # Draft model: a small model that guesses tokens for the big one. It shares the block tables with the big
        # model (same block ids and slots for every position) but has its own KV cache tensors.
        self.draft_model = None
        if config.speculative_method == "draft":
            self.draft_model = Qwen3ForCausalLM(config.draft_hf_config)
            load_model(self.draft_model, config.draft_model)
        self.models = {"target": self.model, "draft": self.draft_model}
        self.sampler = Sampler()
        # widest block table a decode step can need: checking guesses writes up to k positions past max_model_len
        k = config.num_speculative_tokens if config.speculative_method else 0
        self.max_num_blocks = (config.max_model_len + k + self.block_size - 1) // self.block_size
        self.graph_bs = self.make_graph_bs()
        self.mixed_buckets = self.make_mixed_buckets()
        self.warmup_model()
        self.allocate_kv_cache()
        if self.mixed_buckets:
            self.piecewise_vars = self.make_piecewise_vars()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def make_graph_bs(self) -> dict[str, dict[int, list[int]]]:
        """Batch sizes CUDA graphs are recorded for, per model and per number of tokens each sequence brings.
        Known even with enforce_eager, see prepare_decode.
          target, 1 token:    plain decode (upstream)
          target, 1+k tokens: checking k guesses, for each k a step may use (one k, or several with dynamic k)
          draft, 2 tokens:    the draft model's first step (see run_draft)
          draft, 1 token:     the draft model's later steps (only when k >= 2)"""
        config = self.config
        ks = [k for k in config.spec_k_choices if k > 0] if config.speculative_method else []
        graph_bs = {"target": {1: batch_buckets(min(config.max_num_seqs, 512))}, "draft": {}}
        for k in ks:
            graph_bs["target"][1 + k] = batch_buckets(config.spec_batch_limit(k))
        if self.draft_model is not None:
            # 2 tokens per sequence also runs on steps that don't guess, so it goes up to 256 like plain decode's 512 tokens
            sizes = batch_buckets(min(config.max_num_seqs, 256))
            graph_bs["draft"][2] = sizes
            if max(ks) >= 2:
                # later steps run on the batch as padded by the first step, so use the same sizes
                most = max(config.spec_batch_limit(k) for k in ks if k >= 2)
                graph_bs["draft"][1] = sizes[:next(i for i, x in enumerate(sizes) if x >= most) + 1]
        return graph_bs

    def make_mixed_buckets(self) -> list[int]:
        """Total token counts mixed steps are padded to, each with its piecewise CUDA graphs (see run_piecewise):
        every 64 up to max_num_batched_tokens, at most 512; bigger mixed steps run eagerly, unpadded. Known even
        with enforce_eager, which pads the same way so that eager and graphs give the same tokens. Empty without
        mixed batching or with mixed_cuda_graph off (P4a: mixed steps eager, unpadded)."""
        config = self.config
        if not (config.enable_mixed_batching and config.mixed_cuda_graph):
            return []
        most = min(config.max_num_batched_tokens, 512)
        return list(range(64, most, 64)) + [most]

    def make_piecewise_vars(self) -> dict[str, torch.Tensor]:
        """Fixed buffers the piecewise graphs read and write, sized for the biggest bucket and used as [:n]."""
        n = self.mixed_buckets[-1]
        attn = self.model.model.layers[0].self_attn
        hidden_size = self.config.hf_config.hidden_size
        return dict(
            input_ids=torch.zeros(n, dtype=torch.int64),
            positions=torch.zeros(n, dtype=torch.int64),
            q=torch.zeros(n, attn.num_heads, attn.head_dim),
            k=torch.zeros(n, attn.num_kv_heads, attn.head_dim),
            v=torch.zeros(n, attn.num_kv_heads, attn.head_dim),
            o=torch.zeros(n, attn.num_heads, attn.head_dim),    # attention output
            residual=torch.zeros(n, hidden_size),
            hidden=torch.zeros(n, hidden_size),                 # final output (after the last norm)
        )

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.piecewise_graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        if self.config.speculative_method:
            self.warmup_verify()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def warmup_verify(self):
        # Verifying guesses produces logits for every query token, (1+k) rows per request, e.g. 510 x 151936
        # for 102 requests and k=4, and with a draft model its logits for every guess too. The KV cache gets
        # whatever memory is left after the peak seen during warmup, so run the largest verify of every k here.
        V = self.config.hf_config.vocab_size
        for k in self.config.spec_k_choices:
            if k == 0:
                continue
            num_seqs = self.config.spec_batch_limit(k)    # bigger batches don't guess (see LLMEngine.step)
            set_context(False)
            hidden_states = torch.zeros(num_seqs * (1 + k), self.config.hf_config.hidden_size)
            logits = self.model.compute_logits(hidden_states)
            drafts = torch.zeros(num_seqs, k, dtype=torch.int64)
            num_drafts = torch.full((num_seqs,), k, dtype=torch.int64)
            temperatures = torch.ones(num_seqs)
            draft_logits = torch.zeros(num_seqs, k, V) if self.draft_model is not None else None
            verify_drafts(logits, drafts, num_drafts, temperatures, draft_logits)
            del hidden_states, logits, draft_logits
            reset_context()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        shapes = {}    # which model -> shape of one layer's K (or V) cache, per block
        for which, cfg in (("target", hf_config), ("draft", config.draft_hf_config)):
            if self.models[which] is None:
                continue
            num_kv_heads = cfg.num_key_value_heads // self.world_size
            head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
            shapes[which] = (cfg.num_hidden_layers, num_kv_heads, head_dim)
        # with a draft model, one block holds the KV of the same 256 positions for both models
        block_bytes = sum(2 * layers * self.block_size * heads * dim * hf_config.dtype.itemsize for layers, heads, dim in shapes.values())
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_caches = {}
        for which, (layers, heads, dim) in shapes.items():
            kv_cache = torch.empty(2, layers, config.num_kvcache_blocks, self.block_size, heads, dim)
            layer_id = 0
            for module in self.models[which].modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = kv_cache[0, layer_id]
                    module.v_cache = kv_cache[1, layer_id]
                    layer_id += 1
            self.kv_caches[which] = kv_cache

    def prepare_block_tables(self, seqs: list[Sequence], num_padding_seqs: int = 0, width: int = 0):
        max_len = max([width] + [len(seq.block_table) for seq in seqs])
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables += [[-1] * max_len] * num_padding_seqs
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence], num_query_tokens: int = 1, num_last_tokens: int = 1, which: str = "target"):
        # Each sequence brings num_query_tokens tokens: its last num_last_tokens tokens (normally 1: the last token,
        # whose KV isn't cached yet), then for the big model its guessed tokens, then padding. Padding is never
        # accepted; its KV lands past the cached length and is overwritten later. All sequences use the same
        # count so attention can treat q as [seqs, count, ...].
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            tokens = [seq.last_token] if num_last_tokens == 1 else seq.token_ids[len(seq) - num_last_tokens:]
            if which == "target":
                tokens = tokens + seq.draft_token_ids
            tokens += [tokens[-1]] * (num_query_tokens - len(tokens))
            start = len(seq) - num_last_tokens
            end = start + num_query_tokens
            input_ids.extend(tokens)
            positions.extend(range(start, end))
            context_lens.append(end)
            slot_mapping.extend(seq.block_table[i // self.block_size] * self.block_size + i % self.block_size for i in range(start, end))
        # Pad the batch with empty sequences up to a size CUDA graphs are recorded for, also when running eagerly.
        # cuBLAS results depend slightly on the batch size, so this way eager and graph compute exactly the same
        # thing and give the same tokens. Empty sequences write no KV (slot -1) and see nothing (length 0); their
        # outputs are dropped. (Until P5b2 only with speculative decoding: plain decode then ran 10 requests as 10
        # in eager but 16 in a graph, which gave different tokens on Qwen3-8B, whose down_proj GEMM changes kernel
        # between batch 10 and 16; 0.6B's shapes don't at batch <= 16, which hid it.)
        num_padding = 0
        sizes = self.graph_bs[which].get(num_query_tokens)
        if sizes and len(seqs) <= sizes[-1]:
            num_padding = next(x for x in sizes if x >= len(seqs)) - len(seqs)
            input_ids += [0] * (num_padding * num_query_tokens)
            positions += [0] * (num_padding * num_query_tokens)
            slot_mapping += [-1] * (num_padding * num_query_tokens)
            context_lens += [0] * num_padding
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # Also as wide as the CUDA graphs' block tables: with few sequences flash-attn splits each one's KV into
        # chunks computed in parallel, and the number of chunks follows from the table's width, so a narrower
        # table in eager would add the chunks up differently (seen as one flipped bf16 near-tie, P4b)
        block_tables = self.prepare_block_tables(seqs, num_padding, self.max_num_blocks)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, which: str = "target"):
        """Runs the big model (which="target") or the draft model on the current context; returns hidden states.
        With a CUDA graph they live in a buffer the next replay overwrites, so use them before running again."""
        model = self.models[which]
        context = get_context()
        num_tokens = input_ids.size(0)
        bs = 0 if is_prefill else context.context_lens.size(0)
        graph_bs = None if is_prefill or self.enforce_eager else self.graph_bs[which].get(num_tokens // bs)
        if graph_bs is None or bs > graph_bs[-1]:
            return model(input_ids, positions)
        graph = self.graphs[(which, num_tokens // bs, next(x for x in graph_bs if x >= bs))]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:num_tokens] = input_ids
        graph_vars["positions"][:num_tokens] = positions
        graph_vars["slot_mapping"].fill_(-1)    # the graph's unused rows must not write KV
        graph_vars["slot_mapping"][:num_tokens] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return graph_vars["outputs"][which][:num_tokens]

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool, num_spec_tokens: int = 0) -> list[list[int]]:
        """Returns the new tokens of each sequence (one, or several with speculative decoding).
        num_spec_tokens: k for this decode step (0 = plain decode). With n-gram the guesses are already in
        seq.draft_token_ids; with a draft model they are made here."""
        if not is_prefill and num_spec_tokens and (self.draft_model is not None or any(seq.draft_token_ids for seq in seqs)):
            return self.run_verify(seqs, num_spec_tokens)
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        hidden_states = self.run_model(input_ids, positions, is_prefill)
        logits = self.model.compute_logits(hidden_states)
        if self.draft_model is not None:
            # keep the draft model's KV cache in step with the big model's: in prefill it reads the same tokens
            # (same context, so the same slots and the same prefix-cache blocks), in decode see run_draft
            if is_prefill:
                self.run_model(input_ids, positions, True, "draft")
            else:
                self.run_draft(seqs, 0, temperatures)
        if logits is not None and logits.size(0) > len(seqs):    # padding sequences, see prepare_decode
            logits = logits[:len(seqs)]
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return [[token_id] for token_id in token_ids] if self.rank == 0 else None

    @torch.inference_mode()
    def run_draft(self, seqs: list[Sequence], k: int, temperatures: torch.Tensor):
        """The draft model guesses k tokens per sequence, one step at a time. Returns the guesses [S, k] and, if any
        sequence samples (temperature > 0), the draft logits they were sampled from [S, k, V] (else None).

        Between steps the big model has valid KV up to position N-2 for a sequence of N tokens. The draft model
        may lag one position behind: when the last step accepted all k guesses, the k-th guess was output by the
        draft model but never fed into it. So the first step always feeds 2 tokens, positions N-2 and N-1
        (rewriting N-2 if it was there already, with the same values), and takes the guess from the last one.
        Every later step feeds the previous guess. The KV written goes up to position N+k-2, inside the 1+k
        slots the scheduler reserved. Rejected guesses leave KV past the valid length, overwritten later.
        k = 0: only the first step, without logits. Steps that don't guess run it too, so the lag never grows.

        A wrong draft KV cache would not make the output wrong (the big model checks every guess), only the
        acceptance rate lower, so tests must look at the acceptance rate to catch bugs here."""
        S = len(seqs)
        input_ids, positions = self.prepare_decode(seqs, 2, num_last_tokens=2, which="draft")
        hidden_states = self.run_model(input_ids, positions, False, "draft")
        if k == 0:
            reset_context()
            return None, None
        context = get_context()
        B = context.context_lens.size(0)    # with padding sequences
        keep_logits = bool((temperatures > 0).any())
        logits = self.draft_model.compute_logits(hidden_states.view(B, 2, -1)[:S, 1])
        tokens = [self.sampler(logits, temperatures)]
        all_logits = [logits] if keep_logits else None
        if k > 1:
            # later steps: build the inputs on the GPU (no CPU work per step). Padding sequences stay empty.
            real = context.context_lens > 0
            last_pos = positions.view(B, 2)[:, 1]    # N-1
            block_tables = context.block_tables
            input_ids = torch.zeros(B, dtype=torch.int64, device=positions.device)
            for j in range(1, k):
                pos = torch.where(real, last_pos + j, 0)
                block = block_tables.gather(1, (pos // self.block_size).unsqueeze(1)).squeeze(1)
                slot_mapping = torch.where(real, block * self.block_size + pos % self.block_size, -1).to(torch.int32)
                context_lens = torch.where(real, pos + 1, 0).to(torch.int32)
                set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
                input_ids[:S] = tokens[-1]
                hidden_states = self.run_model(input_ids, pos, False, "draft")
                logits = self.draft_model.compute_logits(hidden_states[:S])
                tokens.append(self.sampler(logits, temperatures))
                if keep_logits:
                    all_logits.append(logits)
        reset_context()
        return torch.stack(tokens, dim=1), torch.stack(all_logits, dim=1) if keep_logits else None

    @torch.inference_mode()
    def run_verify(self, seqs: list[Sequence], k: int) -> list[list[int]]:
        """Speculative decoding: run the last token plus the guesses through the model in one pass, keep
        the guesses the model agrees with, and add one token chosen by the model."""
        # Always 1+k tokens per sequence, however many guesses there are: fixed shapes let CUDA graphs be
        # reused, and eager runs the very same shapes so both give identical results.
        temperatures = self.prepare_sample(seqs)
        draft_logits = None
        if self.draft_model is not None:
            drafts, draft_logits = self.run_draft(seqs, k, temperatures)
            for seq, guesses in zip(seqs, drafts.tolist()):
                seq.draft_token_ids = guesses[:seq.max_drafts(k)]
        else:
            drafts = [seq.draft_token_ids + [0] * (k - len(seq.draft_token_ids)) for seq in seqs]
            drafts = torch.tensor(drafts, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True).view(len(seqs), k)
        num_drafts = torch.tensor([len(seq.draft_token_ids) for seq in seqs], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        input_ids, positions = self.prepare_decode(seqs, 1 + k)
        hidden_states = self.run_model(input_ids, positions, False)[:len(seqs) * (1 + k)]    # drop padding sequences
        logits = self.model.compute_logits(hidden_states)
        num_accepted, next_tokens = verify_drafts(logits, drafts, num_drafts, temperatures, draft_logits)
        reset_context()
        return [seq.draft_token_ids[:a] + [t] for seq, a, t in zip(seqs, num_accepted.tolist(), next_tokens.tolist())]

    def prepare_mixed(self, seqs: list[Sequence], k: int):
        b = mixed_batch(seqs, k, self.block_size)
        block_tables = self.prepare_block_tables(seqs)
        def to_gpu(x, dtype):
            return torch.tensor(x, dtype=dtype, pin_memory=True).cuda(non_blocking=True)
        set_context(True, to_gpu(b["cu_seqlens_q"], torch.int32), to_gpu(b["cu_seqlens_k"], torch.int32), b["max_seqlen_q"], b["max_seqlen_k"],
                    to_gpu(b["slot_mapping"], torch.int32), None, block_tables, to_gpu(b["logits_indices"], torch.int64))
        return to_gpu(b["input_ids"], torch.int64), to_gpu(b["positions"], torch.int64)

    @torch.inference_mode()
    def run_mixed(self, seqs: list[Sequence], k: int) -> list[list[int]]:
        """A step from Scheduler.schedule_mixed: decode requests (checking k guesses each, k may be 0) and prefill
        chunks. Returns the new tokens of each request, [] for a prefill chunk that isn't the request's last.
        A step that is all decode or all prefill runs the usual way (so pure decode steps keep their CUDA graphs)."""
        decode_seqs = [seq for seq in seqs if not seq.is_prefill]
        prefill_seqs = [seq for seq in seqs if seq.is_prefill]
        if not prefill_seqs:
            return self.run(seqs, False, k)
        if not decode_seqs:
            token_ids = self.run(seqs, True)
            return [t if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens else [] for seq, t in zip(seqs, token_ids)]
        assert self.draft_model is None and self.world_size == 1
        if not any(seq.draft_token_ids for seq in decode_seqs):
            k = 0    # nothing to check: 1 token per decode request (the scheduler reserved for 1+k, more than enough)
        order = seqs
        seqs = decode_seqs + prefill_seqs    # decode rows come first in the logits
        input_ids, positions = self.prepare_mixed(seqs, k)
        if self.mixed_buckets and input_ids.size(0) <= self.mixed_buckets[-1]:
            hidden_states = self.run_piecewise(input_ids, positions)
        else:
            hidden_states = self.run_model(input_ids, positions, True)
        logits = self.model.compute_logits(hidden_states)
        reset_context()
        D = len(decode_seqs)
        temperatures = self.prepare_sample(seqs)
        if k:
            drafts = [seq.draft_token_ids + [0] * (k - len(seq.draft_token_ids)) for seq in decode_seqs]
            drafts = torch.tensor(drafts, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True).view(D, k)
            num_drafts = torch.tensor([len(seq.draft_token_ids) for seq in decode_seqs], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            num_accepted, next_tokens = verify_drafts(logits[:D * (1 + k)], drafts, num_drafts, temperatures[:D])
            new_tokens = [seq.draft_token_ids[:a] + [t] for seq, a, t in zip(decode_seqs, num_accepted.tolist(), next_tokens.tolist())]
        else:
            new_tokens = [[t] for t in self.sampler(logits[:D], temperatures[:D]).tolist()]
        done = [seq for seq in prefill_seqs if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens]
        first_tokens = iter([])
        if done:
            rows = logits[D * (1 + k):]
            temps = torch.tensor([seq.temperature for seq in done], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
            first_tokens = iter(self.sampler(rows, temps).tolist())
        tokens_of = {id(seq): t for seq, t in zip(decode_seqs, new_tokens)}
        for seq in prefill_seqs:
            tokens_of[id(seq)] = [next(first_tokens)] if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens else []
        return [tokens_of[id(seq)] for seq in order]

    def run_segment(self, i: int, n: int):
        """Segment i of the model for n tokens, on the buffers of make_piecewise_vars. With L layers there are L+1
        segments, split at the attentions: segment 0 = embedding + layer 0 up to its attention; segment i =
        the rest of layer i-1 after its attention + layer i up to its attention; segment L = the rest of the last
        layer + the final norm. Each reads input_ids / positions / o / residual and writes q / k / v / residual (or
        hidden). Nothing here reads get_context(): every token is computed on its own, so a segment only depends
        on n, and one recording serves every mixed step padded to n."""
        m, pv = self.model.model, self.piecewise_vars
        if i == 0:
            hidden_states, residual = m.embed_tokens(pv["input_ids"][:n]), None
        else:
            hidden_states, residual = m.layers[i - 1].post_attn(pv["o"][:n], pv["residual"][:n])
        if i == len(m.layers):
            pv["hidden"][:n] = m.norm(hidden_states, residual)[0]
            return
        q, k, v, residual = m.layers[i].pre_attn(pv["positions"][:n], hidden_states, residual)
        pv["q"][:n] = q
        pv["k"][:n] = k
        pv["v"][:n] = v
        pv["residual"][:n] = residual

    @torch.inference_mode()
    def run_piecewise(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The big model on a mixed step (context set by prepare_mixed): the segments between attentions replay
        CUDA graphs recorded per total token count (eager: run the same code, same padded shapes), the attentions
        run eagerly in between on the real tokens only. Graph replays save the kernel launch overhead of
        everything but attention, which is most of the time of a small step.
        Padding: the step's n tokens are padded to the next bucket with token 0 at position 0. Padded rows never
        enter attention (it gets [:n]: no KV written, not in cu_seqlens), and their attention output is zero.
        Their other outputs are garbage that no real row reads (every op outside attention is per token), and
        logits_indices never points at them. Returns the hidden states of the n real tokens (a buffer the next
        step overwrites)."""
        n = input_ids.size(0)
        bucket = next(x for x in self.mixed_buckets if x >= n)
        pv = self.piecewise_vars
        pv["input_ids"][:n] = input_ids
        pv["input_ids"][n:bucket] = 0
        pv["positions"][:n] = positions
        pv["positions"][n:bucket] = 0
        pv["o"][n:bucket] = 0
        layers = self.model.model.layers
        for i in range(len(layers) + 1):
            if i:
                pv["o"][:n] = layers[i - 1].self_attn.attn(pv["q"][:n], pv["k"][:n], pv["v"][:n], pv["positions"][:n])
            if self.enforce_eager:
                self.run_segment(i, bucket)
            else:
                self.piecewise_graphs[(bucket, i)].replay()
        return pv["hidden"][:n]

    def measure_step_costs(self,context_len: int = 512, repeat: int = 3) -> tuple[dict[int, list[float]], dict[int, list[int]]]:
        """For dynamic k: time one decode step (ModelRunner.run, which includes drafting with a draft model) for
        every k a step may use and every batch-size bucket. Returns (costs, buckets): costs[k][i] is milliseconds
        at batch size buckets[k][i]. Made-up requests of context_len tokens; their KV blocks are real (nothing
        else uses the cache yet) and are reused across requests when there aren't enough, which keeps the amount
        of KV read per step right. k = 0 is only measured up to the largest batch any k > 0 allows: bigger
        batches never guess anyway. Greedy sampling."""
        config = self.config
        ks = config.spec_k_choices
        k_max = max(ks)
        context_len = min(context_len, config.num_kvcache_blocks * self.block_size - k_max - 1)
        blocks_per_seq = (context_len + k_max + self.block_size - 1) // self.block_size
        num_sets = config.num_kvcache_blocks // blocks_per_seq
        most = max(config.spec_batch_limit(k) for k in ks if k > 0)
        sp = SamplingParams(temperature=0)
        costs, buckets = {}, {}
        for k in ks:
            sizes = self.graph_bs["target"][1 + k]
            if k == 0:
                sizes = [x for x in sizes if x < most] + [next(x for x in sizes if x >= most)]
            buckets[k], costs[k] = sizes, []
            for bs in sizes:
                # each bigger batch needs bigger temporaries; without this the allocator keeps every size it has
                # seen cached and GPU memory runs out (see warn_if_low_memory)
                torch.cuda.empty_cache()
                seqs = []
                for i in range(bs):
                    seq = Sequence([100 + i] * context_len, sp)
                    j = i % num_sets
                    seq.block_table = list(range(j * blocks_per_seq, (j + 1) * blocks_per_seq))
                    seq.num_cached_tokens = context_len - 1
                    seqs.append(seq)
                times = []
                for _ in range(1 + repeat):    # the first run warms up
                    for seq in seqs:
                        seq.draft_token_ids = [7] * k if self.draft_model is None else []
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    self.run(seqs, False, k)
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t)
                costs[k].append(1000 * min(times[1:]))
        torch.cuda.empty_cache()
        return costs, buckets

    def warn_if_low_memory(self, min_free_mb: int = 300):
        """Called once setup is done. On WSL / Windows the NVIDIA driver doesn't fail when GPU memory runs out: it
        quietly puts new allocations in system memory, and every step that touches them becomes very slow."""
        free = torch.cuda.mem_get_info()[0] // 2**20
        if free < min_free_mb:
            print(f"nanovllm: only {free} MB of GPU memory free after startup. On WSL / Windows the driver may then use "
                  f"system memory instead and run very slowly; try a lower gpu_memory_utilization "
                  f"(now {self.config.gpu_memory_utilization}).", file=sys.stderr)

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        families = [(which, n, bs) for which, by_n in self.graph_bs.items() for n, sizes in by_n.items() for bs in sizes]
        max_num_tokens = max(n * bs for _, n, bs in families)
        max_num_seqs = max(bs for _, _, bs in families)
        input_ids = torch.zeros(max_num_tokens, dtype=torch.int64)
        positions = torch.zeros(max_num_tokens, dtype=torch.int64)
        slot_mapping = torch.zeros(max_num_tokens, dtype=torch.int32)
        context_lens = torch.zeros(max_num_seqs, dtype=torch.int32)
        block_tables = torch.zeros(max_num_seqs, self.max_num_blocks, dtype=torch.int32)
        outputs = {}
        for which, cfg in (("target", config.hf_config), ("draft", config.draft_hf_config)):
            if self.graph_bs[which]:
                outputs[which] = torch.zeros(max(n * sizes[-1] for n, sizes in self.graph_bs[which].items()), cfg.hidden_size)
        self.graphs = {}
        self.graph_pool = None

        # biggest first: all graphs share one memory pool, sized by the first capture
        for which, n, bs in sorted(families, key=lambda x: x[1] * x[2], reverse=True):
            model, out = self.models[which], outputs[which]
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:n * bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            out[:n * bs] = model(input_ids[:n * bs], positions[:n * bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                out[:n * bs] = model(input_ids[:n * bs], positions[:n * bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[(which, n, bs)] = graph
            torch.cuda.synchronize()
            reset_context()

        # piecewise graphs for mixed steps (see run_piecewise), in the same memory pool. No context is set while
        # recording: a segment that read it would fail here instead of silently recording one step's batch layout.
        self.piecewise_graphs = {}
        for n in reversed(self.mixed_buckets):
            for i in range(len(self.model.model.layers) + 1):
                self.run_segment(i, n)    # warmup
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, self.graph_pool):
                    self.run_segment(i, n)
                self.piecewise_graphs[(n, i)] = graph
            torch.cuda.synchronize()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
