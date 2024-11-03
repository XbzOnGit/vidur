import atexit
import heapq
import json
from typing import List, Optional

from vidur.config import SimulationConfig
from vidur.entities import Cluster
from vidur.events import BaseEvent, RequestArrivalEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.request_generator import RequestGeneratorRegistry
from vidur.scheduler import BaseGlobalScheduler, GlobalSchedulerRegistry
from vidur.entities import Request

logger = init_logger(__name__)


class Simulator:
    def __init__(self, config: SimulationConfig) -> None:
        self._config: SimulationConfig = config

        self._time = 0
        self._terminate = False
        self._time_limit = self._config.time_limit
        self._request_cnt = 0
        self._request_init_list: Optional[List[Request]] = None
        if not self._time_limit:
            self._time_limit = float("inf")

        self._event_queue = []

        self._event_trace = []
        self._event_chrome_trace = []

        self._cluster = Cluster(
            self._config.cluster_config,
            self._config.metrics_config,
            self._config.request_generator_config,
        )
        self._metric_store = MetricsStore(self._config)
        
        self._request_generator = None
        self._jsonl_trace_file = self._config.jsonl_trace_file
        if len(self._jsonl_trace_file) == 0:
            self._request_generator = RequestGeneratorRegistry.get(
                self._config.request_generator_config.get_type(),
                self._config.request_generator_config,
            )
        self._scheduler = GlobalSchedulerRegistry.get(
            self._config.cluster_config.global_scheduler_config.get_type(),
            self._config,
            self._cluster.replicas,
            self,
        )

        self._init_event_queue()
        atexit.register(self._write_output)

    @property
    def scheduler(self) -> BaseGlobalScheduler:
        return self._scheduler

    @property
    def metric_store(self) -> MetricsStore:
        return self._metric_store
    
    def loop_until(self, wait_event: Optional[BaseEvent]) -> float:
        last_event = None
        while self._event_queue and not self._terminate:
            _, event = heapq.heappop(self._event_queue)
            event.set_simulator(self)
            last_event = event
            self._set_time(event._time)
            new_events = event.handle_event(self._scheduler, self._metric_store)
            self._add_events(new_events)

            if self._config.metrics_config.write_json_trace:
                self._event_trace.append(event.to_dict())

            if self._config.metrics_config.enable_chrome_trace:
                chrome_trace = event.to_chrome_trace()
                if chrome_trace:
                    self._event_chrome_trace.append(chrome_trace)

            if wait_event and event == wait_event:
                break
        if wait_event:
            assert wait_event == last_event
        return self._time

    def add_events(self, events: List[BaseEvent]) -> None:
        self._add_events(events)
    
    def inc_request_cnt(self, inc: int) -> None:
        self._request_cnt += inc

    def run(self) -> None:
        logger.info(
            f"Starting simulation with cluster: {self._cluster} and {len(self._event_queue)} requests"
        )
        self.inc_request_cnt(len(self._event_queue))

        self.loop_until(None)

        assert self._scheduler.is_empty() or self._terminate

        logger.info(f"Simulation ended at: {self._time}s")
        # Now only considering initial requests.
        init_req_cnt = len(self._request_init_list)
        thput = init_req_cnt / self._time
        logger.info(f"Throughput: {thput} req/s")
        ttft_sum = sum([request.prefill_completed_at for request in self._request_init_list])
        avg_ttft = ttft_sum / init_req_cnt
        logger.info(f"Average TTFT: {avg_ttft}s")
        avg_quality = sum([request.quality for request in self._request_init_list]) / init_req_cnt
        logger.info(f"Average quality: {avg_quality}")
        avg_hit_len = sum([request.kv_cache_hit_length for request in self._request_init_list]) / init_req_cnt
        logger.info(f"Average hit length: {avg_hit_len}")
        acc_exec_time_dict = self._scheduler.acc_exec_time
        for replica_id, acc_exec_time in acc_exec_time_dict.items():
            logger.info(f"Replica {replica_id} accumulated execution time: {acc_exec_time}")
        

    def _write_output(self) -> None:
        logger.info("Writing output")

        self._metric_store.plot()
        logger.info("Metrics written")

        if self._config.metrics_config.write_json_trace:
            self._write_event_trace()
            logger.info("Json event trace written")

        if self._config.metrics_config.enable_chrome_trace:
            self._write_chrome_trace()
            logger.info("Chrome event trace written")

    def _add_event(self, event: BaseEvent) -> None:
        heapq.heappush(self._event_queue, (event._priority_number, event))

    def _add_events(self, events: List[BaseEvent]) -> None:
        for event in events:
            self._add_event(event)

    def _init_event_queue(self) -> None:
        if len(self._jsonl_trace_file) > 0:
            requests = []
            with open(self._jsonl_trace_file, "r") as f:
                for line in f:
                    req_dict = json.loads(line)
                    total_len = len(req_dict["tokens"])
                    tokens = req_dict["tokens"]
                    arrived_at = req_dict["arrived_at"]
                    num_decode_tokens = req_dict["num_decode_tokens"]
                    # TODO: REMOVE THIS!!
                    num_decode_tokens = 1
                    num_prefill_tokens = total_len - num_decode_tokens
                    assert num_prefill_tokens > 0 and num_decode_tokens > 0
                    request = Request(arrived_at, num_prefill_tokens, num_decode_tokens, tokens, 
                                      0)
                    requests.append(request)
        else:
            requests = self._request_generator.generate()
        self._request_init_list = requests
        for request in requests:
            self._add_event(RequestArrivalEvent(request.arrived_at, request))

    def _set_time(self, time: float) -> None:
        self._time = time
        if self._time > self._time_limit:
            logger.info(
                f"Time limit reached: {self._time_limit}s terminating the simulation."
            )
            self._terminate = True

    def _write_event_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/event_trace.json"
        with open(trace_file, "w") as f:
            json.dump(self._event_trace, f)

    def _write_chrome_trace(self) -> None:
        trace_file = f"{self._config.metrics_config.output_dir}/chrome_trace.json"

        chrome_trace = {"traceEvents": self._event_chrome_trace}

        with open(trace_file, "w") as f:
            json.dump(chrome_trace, f)
