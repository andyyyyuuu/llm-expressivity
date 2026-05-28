from dataclasses import dataclass
from typing import Literal
from nnsight import LanguageModel

@dataclass
class InterventionConfig:
    type: Literal["embed", "layer"]
    layer: int = 0
    prefix_length: int = 5

    def check_valid(self, model: LanguageModel) -> None:
        if self.type == "layer" and \
          (self.layer < 0 or self.layer >= model.config.num_hidden_layers): 
           raise ValueError(f"Layer {self.layer} out of range for {model.config.num_hidden_layers}-layer transformer")