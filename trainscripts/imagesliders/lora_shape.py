import os
import math
import torch
from typing import Optional
from typing_extensions import Literal
from diffusers import PriorTransformer
from safetensors.torch import save_file


LORA_PREFIX_PRIOR = "lora_prior"
UNET_TARGET_REPLACE_MODULE_TRANSFORMER = ["Attention"]
UNET_TARGET_REPLACE_MODULE_CONV = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER + UNET_TARGET_REPLACE_MODULE_CONV
TRAINING_METHODS = Literal[
    "noxattn",  # train all layers except x-attns and time_embed layers
    "innoxattn",  # train all layers except self attention layers
    "selfattn",  # ESD-u, train only self attention layers
    "xattn",  # ESD-x, train only x attention layers
    "full",  #  train all layers
    "xattn-strict", # q and k values
    "noxattn-hspace",
    "noxattn-hspace-last"
]

class LoRAModule(torch.nn.Module):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        if "Linear" in org_module.__class__.__name__:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = torch.nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(lora_dim, out_dim, bias=False)
        elif "Conv" in org_module.__class__.__name__:
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            if self.lora_dim != lora_dim:
                print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)
        self.multiplier = multiplier
        self.org_module = org_module

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )


class LoRANetwork(torch.nn.Module):
    def __init__(self, prior: PriorTransformer, rank: int, alpha: float, train_method: TRAINING_METHODS, multiplier: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.lora_scale = 1
        self.lora_dim = rank
        self.multiplier = multiplier
        self.prior_loras = self.create_modules(prior, train_method)
        print(f"create LoRA for Prior: {len(self.prior_loras)} modules.")
        lora_names = set()
        for lora in self.prior_loras:
            assert lora.lora_name not in lora_names, f"duplicated lora name: {lora.lora_name}"
            lora_names.add(lora.lora_name)
            lora.apply_to()
            self.add_module(lora.lora_name, lora)
        del prior
        torch.cuda.empty_cache()

    def create_modules(self, root_module: torch.nn.Module, train_method: TRAINING_METHODS) -> list:
        loras = []
        for name, module in root_module.named_modules():
            if train_method == "noxattn" or train_method == "noxattn-hspace" or train_method == "noxattn-hspace-last":
                if "attn2" in name or "time_embed" in name:
                    continue
            elif train_method == "innoxattn":
                if "attn2" in name:
                    continue
            elif train_method == "selfattn":
                if "attn1" not in name:
                    continue
            elif train_method == "xattn" or train_method == "xattn-strict":
                if "attn2" not in name:
                    continue
            elif train_method == "full":
                pass
            else:
                raise NotImplementedError(f"train_method: {train_method} is not implemented.")
            if module.__class__.__name__ in DEFAULT_TARGET_REPLACE:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                        if train_method == 'xattn-strict':
                            if 'out' in child_name:
                                continue
                        if train_method == 'noxattn-hspace':
                            if 'mid_block' not in name:
                                continue
                        if train_method == 'noxattn-hspace-last':
                            if 'mid_block' not in name or '.1' not in name or 'conv2' not in child_name:
                                continue
                        lora_name = LORA_PREFIX_PRIOR + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        lora = LoRAModule(lora_name, child_module, self.multiplier, self.lora_dim, self.alpha)
                        loras.append(lora)
        return loras

    def prepare_optimizer_params(self):
        all_params = []
        if self.prior_loras:
            params = []
            [params.extend(lora.parameters()) for lora in self.prior_loras]
            param_data = {"params": params}
            all_params.append(param_data)
        return all_params

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v
        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
            
    def set_lora_slider(self, scale):
        self.lora_scale = scale

    def __enter__(self):
        for lora in self.prior_loras:
            lora.multiplier = 1.0 * self.lora_scale

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.prior_loras:
            lora.multiplier = 0
