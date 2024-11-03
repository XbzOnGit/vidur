from typing import List

from vidur.entities.base_entity import BaseEntity
from vidur.entities.request import Request
from vidur.logger import init_logger

logger = init_logger(__name__)


# a decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


class Batch(BaseEntity):
    def __init__(
        self,
        replica_id: int,
        requests: List[Request],
        num_tokens: List[int],
    ) -> None:
        self._id = Batch.generate_id()
        self._replica_id = replica_id
        self._kv_cache_hit_length_restore = None
        self._num_processed_tokens_restore = None
        self._batch_num_tokens_restore = None

        self._requests = requests
        self._num_tokens = num_tokens
        self._total_num_tokens = sum(num_tokens)
        self._num_prefill_tokens = sum(
            [
                (t if not r.is_prefill_complete else 0)
                for r, t in zip(self.requests, self._num_tokens)
            ]
        )

        self._total_num_tokens_rounded = (self._total_num_tokens + 7) // 8 * 8

        self._scheduled_at = None
        self._completed_at = None
        self._scheduled = False
        self._completed = False

    @property
    def replica_id(self) -> int:
        return self._replica_id

    @property
    def creation_time(self) -> float:
        return self._creation_time

    @property
    def num_tokens(self) -> List[int]:
        return self._num_tokens

    @property
    def total_num_tokens(self) -> int:
        return self._total_num_tokens

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        return self.total_num_tokens - self.num_prefill_tokens

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at

    @property
    @check_completed
    def completed_at(self) -> float:
        return self._completed_at

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def scheduled(self) -> bool:
        return self._scheduled

    @property
    def size(self) -> int:
        return len(self._requests)

    @property
    def requests(self) -> List[Request]:
        return self._requests

    @property
    def request_ids(self) -> List[int]:
        return [request.id for request in self._requests]

    @property
    def all_requests_completed(self) -> bool:
        return all([request.completed for request in self._requests])

    def on_schedule(
        self,
        time: float,
    ) -> None:
        self._scheduled_at = time
        self._scheduled = True

        for request in self._requests:
            request.on_batch_schedule(time)

    def on_batch_end(self, time: float):
        self._completed = True
        self._completed_at = time

        for request, num_tokens in zip(self._requests, self._num_tokens):
            request.on_batch_end(time, num_tokens)

    def reset_on_request_and_num_tokens_change(self):
        self._total_num_tokens = sum(self._num_tokens)
        self._num_prefill_tokens = sum(
            [
                (t if not r.is_prefill_complete else 0)
                for r, t in zip(self.requests, self._num_tokens)
            ]
        )
        self._total_num_tokens_rounded = (self._total_num_tokens + 7) // 8 * 8


    def restore_batch_kv(self, stage_no: int):
        if stage_no == 0:
            return
        else:
            for idx in range(len(self.requests)):
                self.requests[idx].set_kv_cache_hit_length(self._kv_cache_hit_length_restore[idx])
                self.requests[idx].set_num_processed_tokens(self._num_processed_tokens_restore[idx])
                self.num_tokens[idx] = self._batch_num_tokens_restore[idx]
            self.reset_on_request_and_num_tokens_change()

    def modify_batch_kv(self, hit_lens: List[int]):
        kv_cache_hit_length = []
        num_processed_tokens_list = []
        batch_num_tokens_list = []
        for req_bidx, request in enumerate(self.requests):
            kv_cache_hit_length.append(request.kv_cache_hit_length)
            num_processed_tokens_list.append(request.num_processed_tokens)
            batch_num_tokens_list.append(self.num_tokens[req_bidx])
            if not request.is_prefill_complete:
                hit_token_length = hit_lens[req_bidx]
                # NOTE: Now only effective then care.
                total_seq_len = request.num_processed_tokens + self.num_tokens[req_bidx]
                assert hit_token_length <= total_seq_len, f"hit_token_length: {hit_token_length}, total_seq_len: {total_seq_len}"
                if hit_token_length == total_seq_len:
                    hit_token_length -= 1
                if hit_token_length > request.num_processed_tokens:
                    diff_len = hit_token_length - request.num_processed_tokens
                    request.set_kv_cache_hit_length(hit_token_length)
                    request.set_num_processed_tokens(hit_token_length)
                    # Because will not do a full hit.
                    assert hit_token_length < total_seq_len
                    assert diff_len > 0
                    self.num_tokens[req_bidx] -= diff_len
        
        self.reset_on_request_and_num_tokens_change()

        self._kv_cache_hit_length_restore = kv_cache_hit_length
        self._num_processed_tokens_restore = num_processed_tokens_list
        self._batch_num_tokens_restore = batch_num_tokens_list


    @property
    def preempted_requests(self) -> List[Request]:
        return [request for request in self._requests if request.preempted]

    @property
    def completed_requests(self) -> List[Request]:
        return [request for request in self._requests if request.completed]

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "size": self.size,
            "replica_id": self._replica_id,
            "scheduled_at": self._scheduled_at,
            "completed_at": self._completed_at,
            "scheduled": self._scheduled,
            "request_ids": self.request_ids,
            "num_tokens": self._num_tokens,
            "num_prefill_tokens": self.num_prefill_tokens,
            "num_decode_tokens": self.num_decode_tokens,
        }
