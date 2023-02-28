import math
from typing import Optional

import loralib as lora
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraLinear(lora.LoRALayer, nn.Module):
    """Replace in-place ops to out-of-place ops to fit gemini. Convert a torch.nn.Linear to LoraLinear.
    """

    def __init__(
        self,
        weight: nn.Parameter,
        bias: Optional[nn.Parameter],
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
    ):
        nn.Module.__init__(self)
        lora.LoRALayer.__init__(self,
                                r=r,
                                lora_alpha=lora_alpha,
                                lora_dropout=lora_dropout,
                                merge_weights=merge_weights)
        self.weight = weight
        self.bias = bias

        out_features, in_features = weight.shape
        self.in_features = in_features
        self.out_features = out_features

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):

        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Module.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):

        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Module.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):

        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result = result + \
                    (self.lora_dropout(x) @ self.lora_A.t()
                     @ self.lora_B.t()) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


def lora_linear_wrapper(linear: nn.Linear, lora_rank: int) -> LoraLinear:
    assert lora_rank <= linear.in_features, f'LoRA rank ({lora_rank}) must be less than or equal to in features ({linear.in_features})'
    lora_linear = LoraLinear(linear.weight, linear.bias,
                             r=lora_rank, merge_weights=False)
    return lora_linear


def convert_to_lora_recursively(module: nn.Module, lora_rank: int) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, lora_linear_wrapper(child, lora_rank))
        else:
            convert_to_lora_recursively(child, lora_rank)


class LoRAModule(nn.Module):
    """A LoRA module base class. All derived classes should call `convert_to_lora()` at the bottom of `__init__()`.
    This calss will convert all torch.nn.Linear layer to LoraLinear layer.

    Args:
        lora_rank (int, optional): LoRA rank. 0 means LoRA is not applied. Defaults to 0.
        lora_train_bias (str, optional): Whether LoRA train biases.
            'none' means it doesn't train biases. 'all' means it trains all biases. 'lora_only' means it only trains biases of LoRA layers.
            Defaults to 'none'.
    """

    def __init__(self, lora_rank: int = 0, lora_train_bias: str = 'none') -> None:
        super().__init__()
        self.lora_rank = lora_rank
        self.lora_train_bias = lora_train_bias

    def convert_to_lora(self) -> None:
        if self.lora_rank <= 0:
            return
        convert_to_lora_recursively(self, self.lora_rank)
        lora.mark_only_lora_as_trainable(self, self.lora_train_bias)


def lora_state_dict(model, *args,bias: str = 'none',**kargs) :
    my_state_dict = model.state_dict(*args, **kargs)
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError

class LoraModel(LoRAModule):
    """
    Reward model base class.
    Args:
        model (nn.Module): pretrained model.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(self,
                 model: nn.Module,
                 lora_rank: int = 0,
                 lora_train_bias: str = 'none') -> None:
        super().__init__(lora_rank=lora_rank, lora_train_bias=lora_train_bias)
        self.model = model
        self.convert_to_lora()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def state_dict(self,*arg,**kargs):
        return lora_state_dict(self.model, *arg, **kargs)

    def load_state_dict(self, state_dict, *arg, strict=False, **kargs):
        self.model.load_state_dict(state_dict, *arg, strict=False, **kargs)

class Lora128Nb(LoraModel):
    def __init__(self, args, model) -> None:
        super().__init__(model=model, lora_rank=128, lora_train_bias='none')

class SwinLora128Nb(LoraModel):
    default_input_channel  = 71
    default_output_channel = 69 
    pred_channel_for_next_stamp = list(range(2,71))
    def __init__(self, args, model) -> None:
        assert args.subweight
        super().__init__(model=model, lora_rank=128, lora_train_bias='none')