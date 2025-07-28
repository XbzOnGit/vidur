import json
from abc import ABC, abstractmethod
from typing import List

from vidur.config import BaseNetworkGeneratorConfig
from vidur.entities import NetworkChange


class BaseNetworkGenerator(ABC):

    def __init__(self, config: BaseNetworkGeneratorConfig):
        self.config = config

    @abstractmethod
    def generate_network_changes(self) -> List[NetworkChange]:
        pass

    def generate(self) -> List[NetworkChange]:
        net_cs = self.generate_network_changes()
        return net_cs