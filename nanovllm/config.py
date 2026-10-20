import os
import sys
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    speculative_method: str | None = None    # None (off), "ngram" or "draft" (a small model guesses)
    num_speculative_tokens: int = 4          # k: at most k guessed tokens are checked per step
    prompt_lookup_min: int = 2               # ngram: shortest tail to look up in earlier text
    prompt_lookup_max: int = 4               # ngram: longest tail to look up in earlier text
    draft_model: str | None = None           # draft: path of the small model (same tokenizer as the big one)
    draft_hf_config: AutoConfig | None = None
    spec_dynamic_k: bool = False             # choose k each step from the acceptance rate (num_speculative_tokens is the most)
    spec_max_batch_size: int = 0             # guess only when at most this many requests decode together; 0 = 512 // (1+k)
    enable_mixed_batching: bool = False      # one step runs decode requests and prefill chunks together (see Scheduler.schedule_mixed)
    mixed_cuda_graph: bool = True            # mixed batching: mixed steps use piecewise CUDA graphs (ModelRunner.run_piecewise);
                                             # with enforce_eager they run the same padded shapes eagerly. False = P4a (eager, unpadded)
    kernel_backend: str = "compile"          # RMSNorm and SiluAndMul: "compile" (torch.compile) or "triton" (nanovllm/kernels);
                                             # "triton_fused" = triton + q/k norm, rope and KV write in one kernel (kernels/qk_norm_rope_kv.py)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.speculative_method in (None, "ngram", "draft")
        assert self.kernel_backend in ("compile", "triton", "triton_fused")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.speculative_method:
            # drafting needs each request's full token list, which TP>1 doesn't send to the worker processes
            assert self.tensor_parallel_size == 1
            assert self.num_speculative_tokens >= 1
            assert self.max_num_batched_tokens >= 1 + self.num_speculative_tokens
            assert 1 <= self.prompt_lookup_min <= self.prompt_lookup_max
            assert self.spec_max_batch_size >= 0
        if self.speculative_method == "draft":
            assert self.draft_model and os.path.isdir(self.draft_model)
            self.draft_hf_config = AutoConfig.from_pretrained(self.draft_model)
            # the two models must speak the same tokens (Qwen3 models all share one tokenizer)
            assert self.draft_hf_config.vocab_size == self.hf_config.vocab_size
            assert getattr(self.draft_hf_config, "eos_token_id", None) == getattr(self.hf_config, "eos_token_id", None)
            self.max_model_len = min(self.max_model_len, self.draft_hf_config.max_position_embeddings)
            if self.enable_mixed_batching:
                # not supported yet: in a mixed step the draft model would have to write KV for the prefill chunks
                # and guess for the decode requests at once. Fall back to steps that are all prefill or all decode.
                print("nanovllm: enable_mixed_batching is not supported with a draft model, turned off", file=sys.stderr)
                self.enable_mixed_batching = False
        if self.enable_mixed_batching:
            assert self.tensor_parallel_size == 1    # not tested with TP > 1

    @property
    def spec_k_choices(self) -> list[int]:
        """The k values a step may use. Fixed k: just k. Dynamic k: 0 (don't guess), 1, 2, 4, ... and the most k."""
        k = self.num_speculative_tokens
        if not self.spec_dynamic_k:
            return [k]
        return [0] + [x for x in (1, 2, 4, 8, 16, 32) if x < k] + [k]

    def spec_batch_limit(self, k: int) -> int:
        """Most requests a step that checks k guesses may hold. Like plain decode, CUDA graphs cover up to 512
        tokens, and checking feeds 1+k tokens per request; spec_max_batch_size lowers it further if set."""
        limit = min(512 // (1 + k), self.max_num_seqs)
        return min(limit, self.spec_max_batch_size) if self.spec_max_batch_size else limit
