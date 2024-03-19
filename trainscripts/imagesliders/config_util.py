import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from lora import TRAINING_METHODS
from typing import Literal, Optional


PRECISION_TYPES = Literal["fp32", "fp16", "bf16", "float32", "float16", "bfloat16"]
NETWORK_TYPES = Literal["lierla", "c3lier"]


class PretrainedModelConfig(BaseModel):
    name_or_path: str
    v2: bool = False
    v_pred: bool = False

    clip_skip: Optional[int] = None


class NetworkConfig(BaseModel):
    type: NETWORK_TYPES = "lierla"
    rank: int = 4
    alpha: float = 1.0

    training_method: TRAINING_METHODS = "full"


class TrainConfig(BaseModel):
    precision: PRECISION_TYPES = "bfloat16"
    noise_scheduler: Literal["ddim", "ddpm", "lms", "euler_a"] = "ddim"

    iterations: int = 500
    lr: float = 1e-4
    optimizer: str = "adamw"
    optimizer_args: str = ""
    lr_scheduler: str = "constant"

    max_denoising_steps: int = 50


class SaveConfig(BaseModel):
    name: str = "untitled"
    path: str = "./output"
    per_steps: int = 200
    precision: PRECISION_TYPES = "float32"


class LoggingConfig(BaseModel):
    use_wandb: bool = False

    verbose: bool = False


class OtherConfig(BaseModel):
    use_xformers: bool = False


class RootConfig(BaseModel):
    prompts_file: str
    pretrained_model: PretrainedModelConfig

    network: NetworkConfig

    train: Optional[TrainConfig]

    save: Optional[SaveConfig]

    logging: Optional[LoggingConfig]

    other: Optional[OtherConfig]


def parse_precision(precision: str) -> torch.dtype:
    if precision == "fp32" or precision == "float32":
        return torch.float32
    elif precision == "fp16" or precision == "float16":
        return torch.float16
    elif precision == "bf16" or precision == "bfloat16":
        return torch.bfloat16

    raise ValueError(f"Invalid precision type: {precision}")


def load_config_from_yaml(config_path: str) -> RootConfig:
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    root = RootConfig(**config)

    if root.train is None:
        root.train = TrainConfig()

    if root.save is None:
        root.save = SaveConfig()

    if root.logging is None:
        root.logging = LoggingConfig()

    if root.other is None:
        root.other = OtherConfig()

    return root

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1, help="LoRA weight.")
    parser.add_argument("--rank", type=int, default=4, help="Rank of LoRA.")
    parser.add_argument("--name", type=str, default="armsslider", help="Name of the slider.")
    parser.add_argument("--folder_main", type=str, default="datasets/arms", help="The folder to check")
    parser.add_argument( "--test_prompt", type=str, default='a chair', help="prompt for testing")
    parser.add_argument( "--prompts", type=str, default='a chair without arms, a chair with arms', help="prompts for different attribute-scaled images")
    parser.add_argument( "--folders", type=str, default='withoutarms/latents, witharms/latents', help="folders with different attribute-scaled images")
    parser.add_argument( "--scales", type=str, default = '-1, 1', help="scales for different attribute-scaled images")
    parser.add_argument( "--training_method", type=str, default='full', help="training method")
    parser.add_argument( "--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument( "--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument( "--wandb_project", type=str, default="ShapeLoraSliders", help="wandb project name")
    parser.add_argument( "--max_denoising_steps", type=int, default=1024, help="max denoising steps")
    parser.add_argument( "--batch_size", type=int, default=6, help="batch size")
    parser.add_argument( "--guidance_scale", type=float, default=4.0, help="guidance scale")
    parser.add_argument( "--test_steps", type=int, default=50, help="test steps")
    parser.add_argument( "--grad_acc_steps", type=int, default=11, help="max timesteps")
    return parser.parse_args()

def parse_arg(arg: str, is_int: bool = False) -> np.ndarray:
    arg = arg.split(',')
    arg = [f.strip() for f in arg]
    if is_int:
        arg = [int(s) for s in arg]
    return np.array(arg)

def parse_output_path(args: argparse.Namespace) -> Path:
    name = args.name
    name += f'_alpha{args.alpha}'
    name += f'_rank{args.rank}'
    name += f'_{args.training_method}'
    name += f'_lr{args.lr}'
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    name += f'_{current_time}'
    return Path(f'outputs/{name}')
