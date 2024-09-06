from math import ceil
from typing import List
from collections import deque
from vidur.entities.batch import Batch, Request
from vidur.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


# allocate and free by requests.
# only freed when the request is completed or restarted(when preempted).

class VLLMReplicaScheduler(BaseReplicaScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._preempted_requests: List[Request] = []
        self._num_running_batches = 0
        # For vLLM and its derivatives, we only need to set a loose max batch size
        # Memory requirements are handled explicitly by the scheduler
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages
        self._watermark_blocks = int(
            self._config.watermark_blocks_fraction * self._config.num_blocks
        )

    def sub_on_batch_end(self, batch: Batch) -> None:
        self._num_running_batches -= 1

        for request in batch.requests:
            if request.completed:
                self.free(request.id) # Free space managed by this request.
                # Will call kv_controller to free those active blocks.
            else:
                self._preempted_requests.append(request)
                # Requests that has some processed tokens, but not completed.
                # These should be scheduled in another batch.

    def _can_allocate_request(self, request: Request) -> bool:
        # For new request, the first run is prefill, so allocate blocks for it.
        if request.id not in self._allocation_map:
            # new request
            num_required_blocks = ceil(
                (request.num_prefill_tokens) / self._config.block_size
            )
            return (
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )
        # For old request, just make sure that there is at least one more block available.
        # vllm requires at least one block to be available
        return self._config.num_blocks - self._num_allocated_blocks >= 1

    def _allocate_request(self, request: Request) -> None:
        if request.id not in self._allocation_map:
            # new request
            # Allocate for prefill.
            # For vllm scheduler, the first run will finish prefill.
            num_required_blocks = ceil(
                (request.num_prefill_tokens) / self._config.block_size
            )
            self.allocate(request.id, num_required_blocks)
            return

        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size
        # All processed tokens have KV cache.
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)
        assert (
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}"

        if num_tokens_required == 0:
            return
        # Can have one more block for non-prefill ones.
        self.allocate(request.id, 1)

    def cache_reorder(self):
        for request in self._request_queue:
            # FIXME: Now take the first as flag.
            hit_trace = self.get_kv_controller(0).lookup(request, -1.0)
            hit_length = len(hit_trace) - 1
            cached_block_number = hit_length 
            cached_token_number = max(request.num_processed_tokens, cached_block_number * self._config.block_size)
            curent_tokens_after_this = request.num_processed_tokens + self._get_request_next_num_tokens(request)
            if curent_tokens_after_this == request.num_prefill_tokens:
                curent_tokens_after_this += 1
            compute_length = curent_tokens_after_this - request.num_processed_tokens
            assert compute_length > 0
            order_priority = cached_token_number / compute_length
            request.order_priority = order_priority
        # Sort the request queue by order_priority.
        # 0 has the highest priority.
        self._request_queue = deque(sorted(self._request_queue, key=lambda x: x.order_priority, reverse=True))

    def _get_next_batch(self) -> Batch:
        requests = []
        num_tokens = []
        num_batch_tokens = 0
        if self._cache_reordering:
            self.cache_reorder()
        # _request_queue is in order of arrival.
        while self._request_queue:
            request = self._request_queue[0]

            next_num_tokens = self._get_request_next_num_tokens(request)

            if not self._can_allocate_request(request):
                # print(f"Out of loop one cos not can_allocate_request.")
                break

            new_num_tokens = num_tokens + [next_num_tokens]
            new_num_batch_tokens = len(new_num_tokens) * max(new_num_tokens) # Pad to max tokens in batch.
            # Make more constraints on KV cache size used here.
            # Original vidur does not have inter-request KV cache, so kv_size is 0 on prefill.

            if new_num_batch_tokens > self._config.max_tokens_in_batch:
                # print(f"Out of loop one cos new_num_batch_tokens: {new_num_batch_tokens} > max_tokens_in_batch: {self._config.max_tokens_in_batch}")
                break

            if len(self._allocation_map) == self._config.batch_size_cap:
                # How many requests are allocated at the same time in total.
                # print(f"Out of loop one cos len(self._allocation_map): {len(self._allocation_map)} == batch_size_cap: {self._config.batch_size_cap}")
                break

            if len(requests) == self._max_micro_batch_size:
                # How many requests in the current batch.
                # print(f"Out of loop one cos len(requests): {len(requests)} == max_micro_batch_size: {self._max_micro_batch_size}")
                break

            request = self._request_queue.popleft()

            self._allocate_request(request)
            requests.append(request)
            num_tokens.append(next_num_tokens)
            num_batch_tokens += next_num_tokens
        # if not self._request_queue:
            # print(f"Out of loop one cos no more requests in queue.")
        if requests:
            return Batch(self._replica_id, requests, num_tokens)
        # If there is something in request queue, schedule it.
        # else check preempted requests.
        # It prefers those not computed at all.

        # Safer to sort preempted_requests to maintain FIFO order
        self._preempted_requests.sort(key=lambda r: r.arrived_at)
        # all preempted_requests will have prefill completed

        # Here it is batching preempted requests to max_micro_batch_size or 
        # a later request is not able to allocate blocks even if all the preempted requests
        # later freed.
        while self._preempted_requests:
            if len(requests) == self._max_micro_batch_size:
                # print(f"Out of loop outter two cos len(requests): {len(requests)} == max_micro_batch_size: {self._max_micro_batch_size}")
                break

            request = self._preempted_requests.pop(0)
            # For preempted requests, prefill should be finished.
            assert request.id in self._allocation_map
            # So just checking if one more block is available.
            # If yes, allocate request, which can result in one more block allocated or not.
            while not self._can_allocate_request(request):
                # Free up space for the selected one if possible.
                if self._preempted_requests:
                    victim_request = self._preempted_requests.pop(-1)
                    victim_request.restart()
                    self.free(victim_request.id)
                    self._request_queue.appendleft(victim_request)
                else:
                    request.restart()
                    self.free(request.id)
                    self._request_queue.appendleft(request)
                    # print(f"Out of loop inner two cos no more preempted_requests and free itself.")
                    break
                # Here it also adds to request queue, in a way keeping arrival order.
                # preempted_requests is in order of arrival after sort.
                # then popping eariler one should append to left.
            else:
                # Only executed when while condition becomes false.
                # When break/exception, not executed.
                # print(f"Out of loop inner two cos can_allocate_request.")
                self._allocate_request(request)
                next_num_tokens = self._get_request_next_num_tokens(request)
                requests.append(request)
                num_tokens.append(next_num_tokens)

        if not requests:
            # print("No requests to schedule.")
            return

        return Batch(self._replica_id, requests, num_tokens)
