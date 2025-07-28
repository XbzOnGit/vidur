import logging
from typing import List

from vidur.config import SyntheticNetworkGeneratorConfig
from vidur.entities import NetworkChange
from vidur.network_generator.base_network_generator import BaseNetworkGenerator
from vidur.request_generator.request_interval_generator_registry import RequestIntervalGeneratorRegistry

logger = logging.getLogger(__name__)


# TODO: Now empty.
class SyntheticNetworkGenerator(BaseNetworkGenerator):
    def __init__(self, config: SyntheticNetworkGeneratorConfig):
        super().__init__(config)
        """
        self.nc_interval_generator = RequestIntervalGeneratorRegistry.get(
            self.config.interval_generator_config.get_type(),
            self.config.interval_generator_config,
        )
        self.nc_bandwidth_factor_generator = RequestIntervalGeneratorRegistry.get(
            self.config.bandwidth_factor_generator_config.get_type(),
            self.config.bandwidth_factor_generator_config,
        )
        """
        # Now forced to empty.
        assert config.num_net_cs == 0, "Now Synthetic Network Changes must be empty."
        logger.info(f"Generated {config.num_net_cs} network changes.")
    
    def generate_network_changes(self) -> List[NetworkChange]:
        return []