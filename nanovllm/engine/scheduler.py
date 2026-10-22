from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.max_model_len = config.max_model_len
        self.block_size = config.kvcache_block_size
        # speculative decoding: each decode step checks up to k guessed tokens, so reserve KV room for 1+k tokens
        self.num_spec_tokens = config.num_speculative_tokens if config.speculative_method else 0
        self.spec_k_choices = config.spec_k_choices if config.speculative_method else [0]
        # mixed batching: picks k for the decode requests of a step and makes the n-gram guesses, given the most k
        # the step can afford (set by LLMEngine; None = never guess)
        self.choose_num_spec_tokens = None
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_preemptions = 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        num_tokens = 1 + self.num_spec_tokens
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            if self.num_spec_tokens and num_batched_tokens + num_tokens > self.max_num_batched_tokens:
                break
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq, num_tokens):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = num_tokens
                seq.is_prefill = False
                self.block_manager.may_append(seq, num_tokens)
                num_batched_tokens += num_tokens
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def schedule_mixed(self) -> tuple[list[Sequence], int]:
        """enable_mixed_batching: one step holds decode requests and prefill chunks together (like vLLM V1).
        Returns (seqs, k): the decode requests first (seq.is_prefill False, 1+k tokens each: the last token and k
        guess slots), then the prefill chunks (seq.is_prefill True, seq.num_scheduled_tokens tokens each).
          1. Every running request decodes, 1 token each so far; KV blocks for it are taken now, preempting the
             newest running requests if they run out (as in schedule).
          2. k is chosen now, before anything else gets budget or blocks, and only among the k values whose
             1+k tokens per request fit in both the token budget and the free KV blocks. So guessing never
             causes a preemption, and blocks are reserved for the k actually used, not for the largest k.
          3. What is left of max_num_batched_tokens goes to waiting requests, any of which may be cut into chunks:
             first those whose prefill is under way (they already hold all their blocks), then new ones in arrival
             order. New requests only get blocks that are free: never preempt a decoding request to start a new
             one. No new requests on a step that just preempted (the blocks are short).
        Preemption, when a decoding request needs a block and none is free, takes the newest request holding blocks:
        a half-done prefill first (like vLLM V1, where those count as running), then the newest running request.
        Without step 3's order, a request preempted to the front of waiting could stand forever before a half-done
        prefill holding the blocks it needs, with nothing scheduled at all (the P4a deadlock)."""
        budget = self.max_num_batched_tokens
        decode_seqs = []
        preempted = False
        while self.running and len(decode_seqs) < min(self.max_num_seqs, budget):
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq, 1):
                preempted = True
                prefilling = [s for s in self.waiting if s.block_table]
                if prefilling:
                    self.preempt_prefill(prefilling[-1])
                elif self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.block_manager.may_append(seq, 1)
                decode_seqs.append(seq)
        k = 0
        if decode_seqs and self.choose_num_spec_tokens is not None:
            k = self.choose_num_spec_tokens(decode_seqs, self.max_affordable_k(decode_seqs, budget))
        for seq in decode_seqs:
            self.block_manager.may_append(seq, 1 + k)
            seq.num_scheduled_tokens = 1 + k
            seq.is_prefill = False
        budget -= len(decode_seqs) * (1 + k)
        self.running.extendleft(reversed(decode_seqs))

        prefill_seqs = []
        # half-done prefills first, then new requests (only if nothing was preempted, or everything was: then all
        # blocks are free again and starting a new request is what keeps the step from being empty)
        prefilling = [seq for seq in self.waiting if seq.block_table]
        new = [] if preempted and decode_seqs else [seq for seq in self.waiting if not seq.block_table]
        for seq in prefilling + new:
            if budget == 0 or len(decode_seqs) + len(prefill_seqs) == self.max_num_seqs:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break    # arrival order: don't let a later, smaller request jump ahead
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(seq.num_tokens - seq.num_cached_tokens, budget)
            budget -= seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
            prefill_seqs.append(seq)
        assert decode_seqs or prefill_seqs or self.is_finished(), "a request needs more KV blocks than there are"
        return decode_seqs + prefill_seqs, k

    def max_affordable_k(self, seqs: list[Sequence], budget: int) -> int:
        """Largest k a step may use for these decode requests (each already holds blocks for 1 token): 1+k tokens
        each must fit in the token budget and the extra KV blocks in the free ones."""
        for k in sorted(self.spec_k_choices, reverse=True):
            if k == 0:
                break
            if len(seqs) * (1 + k) <= budget and sum(self.block_manager.num_blocks_to_append(seq, 1 + k) for seq in seqs) <= len(self.block_manager.free_block_ids):
                return k
        return 0

    def preempt(self, seq: Sequence):
        self.num_preemptions += 1
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def preempt_prefill(self, seq: Sequence):
        """Drop the blocks of a request whose prefill is half done; it keeps its place in waiting and starts over
        (its finished blocks are hashed, so the prefix cache may still give them back)."""
        self.num_preemptions += 1
        self.block_manager.deallocate(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[list[int]]) -> int:
        """token_ids[i] are the new tokens of seqs[i]: one token normally, or with speculative decoding
        the accepted guesses followed by one token from the model; for a prefill chunk that isn't the last
        one, nothing (ignored). Returns how many tokens were appended.
        Each request is handled by its own kind: seq.is_prefill is set until the scheduler first runs it as a
        decode request (in a step that is all prefill or all decode, every request has the step's kind)."""
        num_appended_total = 0
        for seq, new_token_ids in zip(seqs, token_ids):
            seq.draft_token_ids = []
            is_prefill = seq.is_prefill
            if is_prefill:
                self.block_manager.hash_blocks(seq)
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                if seq.num_cached_tokens < seq.num_tokens:
                    continue
            num_appended = 0
            for token_id in new_token_ids:
                seq.append_token(token_id)
                num_appended += 1
                # one step can produce several tokens, so EOS or a length limit may be hit midway: drop the rest.
                # Stopping at max_model_len keeps block tables within the size CUDA graphs were recorded for.
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens \
                        or seq.num_tokens == self.max_model_len:
                    seq.status = SequenceStatus.FINISHED
                    break
            num_appended_total += num_appended
            if not is_prefill:
                # Valid KV this step: the old last token and the guesses that were kept (all appended tokens
                # except the newest one, whose KV is computed next step). Moving num_cached_tokens forward by
                # exactly that much is the whole "rollback": KV written for rejected guesses lies past it and
                # gets overwritten next step. Hash only after appending, so blocks are hashed from final tokens.
                seq.num_scheduled_tokens = num_appended
                self.block_manager.hash_blocks(seq)
                seq.num_cached_tokens += num_appended
                seq.num_scheduled_tokens = 0
            if seq.is_finished:
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return num_appended_total
