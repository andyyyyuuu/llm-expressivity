# Run module.py to check the default model structure

import torch
from nnsight import LanguageModel
from abc import abstractmethod, ABC


class DownstreamModule(torch.nn.Module, ABC):
    """Base class for downstream modules"""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward pass not implemented")
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError("Output distribution vocab size not implemented")
    
    @property
    @abstractmethod
    def device(self) -> int:
        raise NotImplementedError("Device not implemented")
    
    @property
    @abstractmethod
    def input_shape(self) -> torch.Size:
        raise NotImplementedError("Input shape not set")
    
    @abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError("__repr__ not implemented")
    

    def validate_input(self, x: torch.Tensor) -> None:
        """Generic validation for input tensor matching input_shape"""
        if x.shape != self.input_shape:
            raise ValueError(f"Input tensor must be of shape {self.input_shape}, got {x.shape}")

    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_input(x)
        return self.forward(x)
    

class LMIntervention(DownstreamModule, ABC):
    """Base class for interventions that need to load and trace model output"""

    def __init__(self, model_name: str="meta-llama/Llama-3.2-1B", prefix_length: int=5):
        super().__init__()
        self.model = LanguageModel(model_name, device_map="auto", dispatch=True)
        self.prefix_length = prefix_length
    
    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    @property
    def device(self) -> torch.device:
        return self.model.device

    @property
    def input_shape(self) -> torch.Size:
        return (self.prefix_length, self.model.config.hidden_size)

    @abstractmethod
    def intervene(self, x: torch.Tensor) -> None:
        raise NotImplementedError("Intervention not implemented")
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dummy_input_ids = torch.zeros((1, self.prefix_length), dtype=torch.long, device=self.model.device)
        with self.model.trace(dummy_input_ids):
            self.intervene(x.unsqueeze(0))
            logits = self.model.output.save() # (1, L, V)
        final_logits = logits.logits.squeeze(0)[-1, :] # (V)
        log_probs = torch.log_softmax(final_logits, dim=-1)
        return log_probs



class LayerIntervention(LMIntervention):
    """Between layers"""

    def __init__(self, layer: int, model_name: str="meta-llama/Llama-3.2-1B", prefix_length: int=5):
        super().__init__(model_name, prefix_length=prefix_length)
        if layer < 0 or layer >= self.model.config.num_hidden_layers:
            raise ValueError(f"Layer {layer} is not valid for model {model_name}")
        self.layer = layer
    
    def intervene(self, x: torch.Tensor) -> None:
        self.model.model.layers[self.layer].output[0] = x
    
    def __repr__(self) -> str:
        return f"LayerIntervention(layer={self.layer})"



class EmbedIntervention(LMIntervention):
    """In place of embeddings, before first layer"""

    def __init__(self, model_name: str="meta-llama/Llama-3.2-1B", prefix_length: int=5):
        super().__init__(model_name, prefix_length=prefix_length)
    
    def intervene(self, x: torch.Tensor) -> None:
        self.model.model.embed_tokens.output = x
    
    def __repr__(self) -> str:
        return f"EmbedIntervention()"


class PostNormIntervention(LMIntervention):
    """Post-RMSNorm final bits: unembed, softmax"""
    
    def __init__(self, model_name: str="meta-llama/Llama-3.2-1B", prefix_length: int=5):
        super().__init__(model_name, prefix_length=prefix_length)
    
    def intervene(self, x: torch.Tensor) -> None:
        self.model.model.norm.output = x
    
    def __repr__(self) -> str:
        return f"PostNormIntervention()"
    


if __name__ == "__main__":
    print(LMIntervention().model.model)
