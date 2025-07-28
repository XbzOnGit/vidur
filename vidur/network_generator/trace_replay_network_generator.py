import logging
from typing import List

import pandas as pd

from vidur.config import TraceNetworkGeneratorConfig
from vidur.entities import NetworkChange
from vidur.network_generator.base_network_generator import BaseNetworkGenerator

logger = logging.getLogger(__name__)

class TraceReplayNetworkGenerator(BaseNetworkGenerator):
    """
    Reads a trace csv file containing network change arrival time, its bandwidth factor, pipeline stage id and communication type id to 
    generate network changes.
    """
    def __init__(self, config: TraceNetworkGeneratorConfig):
        super().__init__(config)
        
        # load into a pd dataframe
        self.trace_df = pd.read_csv(config.trace_file)
        
        # scale bandwidth
        self.trace_df["bandwidth_factor"] = (
            self.trace_df["bandwidth_factor"] * config.bandwidth_scale_factor
        )
        
        self.trace_df["bandwidth_factor"] = \
            self.trace_df["bandwidth_factor"].clip(upper=config.max_bandwidth_factor)
        
        # assert still > 0.0
        assert (self.trace_df["bandwidth_factor"] > 0.0).all()
        
        # rescale the time
        self.trace_df["arrived_at"] = (
            self.trace_df["arrived_at"] * config.time_scale_factor
        )
        
        
        logger.info(
            f"Loaded trace file {config.trace_file} with {len(self.trace_df)} network changes"
        )
        
    def generate_network_changes(self):
        net_cs = []
        
        for _, row in self.trace_df.iterrows():
            # Convert into ints.
            net_c = NetworkChange(
                arrived_at=row["arrived_at"],
                pipeline_stage=row["pipeline_stage"].astype(int),
                comm_id=row["comm_id"].astype(int),
                new_bandwidth_factor=row["bandwidth_factor"]
            )
            
            net_cs.append(net_c)
        
        return net_cs
    