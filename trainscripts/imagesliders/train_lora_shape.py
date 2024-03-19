import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"
import torch
import wandb
import random
import argparse
import train_util
import config_util
import numpy as np
from tqdm import tqdm
from pathlib import Path
import lora_shape as lora
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_gif


def flush():
    torch.cuda.empty_cache()
    gc.collect()
    
def text_embedding(tokenizer, text_encoder, device, batch_size, prompt):
    inputs = tokenizer([prompt, prompt], padding=True, return_tensors="pt").to(device)
    outputs = text_encoder(**inputs)
    return outputs.text_embeds.repeat_interleave(batch_size, dim=0)

def train(
    args: argparse.Namespace, 
    device: torch.device,
    prompts: np.ndarray, 
    folders: np.ndarray, 
    scales: np.ndarray, 
    output_path: Path, 
    dtype: torch.dtype = torch.bfloat16
    ):
    
    models_path = output_path / "models"
    models_path.mkdir(parents=True, exist_ok=True)
    results_path = output_path / "results"
    results_path.mkdir(parents=True, exist_ok=True)
    
    wandb.init(project=args.wandb_project, config=args)
    
    pipe = DiffusionPipeline.from_pretrained("openai/shap-e", torch_dtype=dtype).to(device)
    network = lora.LoRANetwork(
        pipe.prior, 
        args.rank, 
        args.alpha, 
        args.training_method
        ).to(device, dtype=dtype)
    optimizer = torch.optim.Adam(network.prepare_optimizer_params(), args.lr)  
    criteria = torch.nn.MSELoss()
    num_embeddings = pipe.prior.config.num_embeddings
    embedding_dim = pipe.prior.config.embedding_dim
    
    pbar = tqdm(range(args.epochs))
    for i in pbar:  
        
        loss_high_for_epoch, loss_low_for_epoch = 0, 0
        
        for _ in range(args.grad_acc_steps):
         
            scale_high = abs(random.choice(list(scales)))
            scale_low = -scale_high
            prompt_high = prompts[scales==scale_high][0]
            prompt_low = prompts[scales==scale_low][0]
            folder_high = folders[scales==scale_high][0]
            folder_low = folders[scales==scale_low][0]
            lats = os.listdir(f'{args.folder_main}/{folder_low}/')
            lats = [lat_ for lat_ in lats if '.pt' in lat_]
            batch_lats = random.sample(lats, args.batch_size)
            lat_high = torch.cat([torch.load(f'{args.folder_main}/{folder_high}/{lat}') for lat in batch_lats])
            lat_low = torch.cat([torch.load(f'{args.folder_main}/{folder_low}/{lat}') for lat in batch_lats])
        
            pipe.scheduler.set_timesteps(args.max_denoising_steps, device)
            timesteps_to = torch.randint(1, args.max_denoising_steps - 1, (1,)).item()
            
            with torch.no_grad():
                seed = random.randint(0, 2*15)
                denoised_latents_high, high_noise = train_util.get_noisy_latent(
                    lat_high, 
                    torch.manual_seed(seed), 
                    pipe.prior, 
                    pipe.scheduler, 
                    timesteps_to, 
                    dtype, 
                    num_embeddings, 
                    embedding_dim
                    )
                denoised_latents_low, low_noise = train_util.get_noisy_latent(
                    lat_low, 
                    torch.manual_seed(seed), 
                    pipe.prior, 
                    pipe.scheduler, 
                    timesteps_to, 
                    dtype, 
                    num_embeddings, 
                    embedding_dim
                    )
                       
            pipe.scheduler.set_timesteps(1000)
            current_timestep = pipe.scheduler.timesteps[int(timesteps_to * 1000 / args.max_denoising_steps)]
            
            network.set_lora_slider(scale=scale_high)
            high_text_embeddings = text_embedding(pipe.tokenizer, pipe.text_encoder, device, args.batch_size, prompt_low)
            with network:
                target_latents_high = train_util.predict_noise_shape(
                    pipe.prior, 
                    pipe.scheduler, 
                    current_timestep, 
                    denoised_latents_high, 
                    high_text_embeddings, 
                    args.guidance_scale
                    ).to("cpu", dtype=torch.float32)
            loss_high = criteria(target_latents_high, high_noise.cpu().to(torch.float32))
            loss_high.backward()
            loss_high_for_epoch += loss_high.item()
            
            low_text_embeddings = text_embedding(pipe.tokenizer, pipe.text_encoder, device, args.batch_size, prompt_high)
            network.set_lora_slider(scale=scale_low)
            with network:
                target_latents_low = train_util.predict_noise_shape(
                pipe.prior, 
                pipe.scheduler, 
                current_timestep, 
                denoised_latents_low, 
                low_text_embeddings, 
                args.guidance_scale
                ).to("cpu", dtype=torch.float32)
            loss_low = criteria(target_latents_low, low_noise.cpu().to(torch.float32))
            loss_low.backward()
            loss_low_for_epoch += loss_low.item()
                
        scaled_loss_high, scaled_loss_low = loss_high_for_epoch * 1000, loss_low_for_epoch * 1000
        log_data = {"loss_high": scaled_loss_high, "loss_low": scaled_loss_low, "iteration": i}
        pbar.set_description(f"loss_high: {scaled_loss_high:.4f}, loss_low: {scaled_loss_low:.4f}")
        optimizer.step()
        network.zero_grad()
        del (target_latents_low, target_latents_high, low_text_embeddings, high_text_embeddings)
        flush()
        
        if i % args.test_steps == 0:
            test_scales = scales.tolist() + [0]
            for scale in test_scales:
                network.set_lora_slider(scale)
                images = pipe(args.test_prompt).images
                result_path = results_path / f"{i}_{scale}.gif"
                log_data[f"scale_{scale}"] = wandb.Video(export_to_gif(images[0], result_path.as_posix()))
        
        wandb.log(log_data)
        
    network.save_weights(models_path / "last.pt", dtype=dtype)

if __name__ == "__main__":
    args = config_util.parse_args()
    output_path = config_util.parse_output_path(args)
    print("output_path: ", output_path)
    prompts = config_util.parse_arg(args.prompts)
    folders = config_util.parse_arg(args.folders)
    scales = config_util.parse_arg(args.scales, is_int=True)
    assert prompts.shape[0] == scales.shape[0], "The number of prompts and scales must be the same."
    assert folders.shape[0] == scales.shape[0], "The number of folders and scales must be the same."
    assert scales.shape[0] == 2, "Currently the number of scales must be 2."
    print("args: ", args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    train(args, device, prompts, folders, scales, output_path)
