from vidur.network_generator.trace_replay_network_generator import (
    TraceReplayNetworkGenerator,
)
from vidur.network_generator.synthetic_network_generator import (
    SyntheticNetworkGenerator
)
from vidur.types import NetworkGeneratorType
from vidur.utils.base_registry import BaseRegistry

class NetworkGeneratorRegistry(BaseRegistry):
    pass

NetworkGeneratorRegistry.register(
    NetworkGeneratorType.SYNTHETIC, SyntheticNetworkGenerator
)
NetworkGeneratorRegistry.register(
    NetworkGeneratorType.TRACE_REPLAY, TraceReplayNetworkGenerator
)
