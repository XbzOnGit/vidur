from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler

class LocalityGlobalScheduler(BaseGlobalScheduler):
    # NOTE: We are distributing requests IMMEDIATELY to the replica.
    # When qps is too high, it will even get lower, cos locality is not so clear on arrival.
    # TODO: Modify the structure of vidur to make lower ones query from higher levels?
    def _find_min_pending_num(self, pending_requests_map: dict, locality_map: dict, locality_target_num):
        min_pending_num = float("inf")
        min_pending_id = -1
        for replica_id, pending_num in pending_requests_map.items():
            if locality_target_num < 0 or locality_map[replica_id] == locality_target_num:
                if pending_num < min_pending_num:
                    min_pending_num = pending_num
                    min_pending_id = replica_id
        assert min_pending_num >= 0
        assert min_pending_id >= 0
        return min_pending_num, min_pending_id
    
    def _find_max_pending_num(self, pending_requests_map: dict, locality_map: dict, locality_target_num):
        max_pending_num = -1
        max_pending_id = -1
        for replica_id, pending_num in pending_requests_map.items():
            if locality_target_num < 0 or locality_map[replica_id] == locality_target_num:
                if pending_num > max_pending_num:
                    max_pending_num = pending_num
                    max_pending_id = replica_id
        assert max_pending_num >= 0
        assert max_pending_id >= 0
        return max_pending_num, max_pending_id

    def _find_max_locality(self, pending_requests_map: dict, locality_map: dict, pending_num_target):
        max_locality = -1
        max_locality_id = -1
        for replica_id, pending_num in pending_requests_map.items():
            if pending_num_target < 0 or pending_num == pending_num_target:
                locality = locality_map[replica_id]
                if locality > max_locality:
                    max_locality = locality
                    max_locality_id = replica_id
        assert max_locality >= 0
        assert max_locality_id >= 0
        return max_locality, max_locality_id

        
    def schedule(self) -> List[Tuple[int, Request]]:
        self.sort_requests()

        pending_requests_map = {
            replica_scheduler.replica_id: replica_scheduler.num_pending_requests
            for replica_scheduler in self._replica_schedulers.values()
        }

        threshold_of_imbalance_ratio = self._config.cluster_config.global_scheduler_config.threshold_of_imbanlance_ratio
        request_mapping = []
        while self._request_queue:
            request = self._request_queue.pop(0)
            locality_map = {
                replica_scheduler.replica_id: replica_scheduler.locality_check(request)
                for replica_scheduler in self._replica_schedulers.values()
            }
            min_pending_num, min_pending_id = self._find_min_pending_num(pending_requests_map, locality_map, -1)
            assert min_pending_num >= 0
            the_chose_id = None
            if min_pending_num == 0:
                _, the_chose_id = self._find_max_locality(pending_requests_map, locality_map, 0)
            else:
                max_pending_num, max_pending_id = self._find_max_pending_num(pending_requests_map, locality_map, -1)
                if max_pending_num / min_pending_num > threshold_of_imbalance_ratio:
                    _, the_chose_id = self._find_max_locality(pending_requests_map, locality_map, min_pending_num)
                else:
                    max_locality, max_locality_id = self._find_max_locality(pending_requests_map, locality_map, -1)
                    assert max_locality >= 0
                    assert max_locality_id >= 0
                    _, the_chose_id = self._find_min_pending_num(pending_requests_map, locality_map, max_locality)
            assert the_chose_id is not None
            assert the_chose_id >= 0
            request_mapping.append((the_chose_id, request))
            # Update pending
            pending_requests_map[the_chose_id] += 1
            # Do not update locality.
        return request_mapping