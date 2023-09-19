
from simple_parsing.helpers import Serializable
from dataclasses import dataclass


@dataclass
class Config(Serializable):
    def get(self, attribute, default=None):
        return getattr(self, attribute, default)