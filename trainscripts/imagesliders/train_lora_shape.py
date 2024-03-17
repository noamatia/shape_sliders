import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_API_KEY"] = "7b14a62f11dc360ce036cf59b53df0c12cd87f5a"
import torch
import wandb
import random
import argparse
import train_util
import model_util
import config_util
import numpy as np
from tqdm import tqdm
from pathlib import Path
import lora_shape as lora


def flush():
    torch.cuda.empty_cache()
    gc.collect()
    
def empty_text_embedding(tokenizer, text_encoder, device, batch_size):
    inputs = tokenizer(["", ""], padding=True, return_tensors="pt").to(device)
    outputs = text_encoder(**inputs)
    return outputs.text_embeds.repeat_interleave(batch_size, dim=0)

def train(
    args: argparse.Namespace, 
    device: torch.device, 
    folders: np.ndarray, 
    scales: np.ndarray, 
    save_path: Path, 
    dtype: torch.dtype = torch.bfloat16
    ):
    
    wandb.init(project=args.wandb_project, config=args)
    
    # TODO: Is it the right way to load the model?
    prior, tokenizer, text_encoder, noise_scheduler, shap_e_renderer = model_util.load_shape_model(device, dtype)
    network = lora.LoRANetwork(
        prior, 
        args.rank, 
        args.alpha, 
        args.training_method
        ).to(device, dtype=dtype)
    optimizer = torch.optim.Adam(network.prepare_optimizer_params(), args.lr)  
    criteria = torch.nn.MSELoss()
    num_embeddings = prior.config.num_embeddings
    embedding_dim = prior.config.embedding_dim
    
    test_noise = torch.randn((1, num_embeddings * embedding_dim), device=device, dtype=dtype, layout=torch.strided)
    test_text_embeddings = empty_text_embedding(tokenizer, text_encoder, device, 1)
    
    data = {}
    for scale in scales:
        data[scale] = []
        folder = folders[scales==scale][0]
        lats = os.listdir(f'{args.folder_main}/{folder}')
        lats = [lat_ for lat_ in lats if '.pt' in lat_]
        for i in range(0, len(lats), args.batch_size):
            batch_lats = lats[i:i+args.batch_size]
            if len(batch_lats) < args.batch_size:
                batch_lats += lats[:args.batch_size - len(batch_lats)]
            assert len(batch_lats) == args.batch_size
            batch_lats = torch.cat([torch.load(f'{args.folder_main}/{folder}/{lat}') for lat in batch_lats])
            data[scale].append(batch_lats)
    
    pbar = tqdm(range(args.epochs))
    for i in pbar:  
         
        scale_to_look = abs(random.choice(list(scales)))
        loss_high_for_epoch, loss_low_for_epoch = 0, 0
        
        for lat1, lat2 in zip(data[-scale_to_look], data[scale_to_look]):
    
            noise_scheduler.set_timesteps(args.max_denoising_steps, device)
            timesteps_to = torch.randint(1, args.max_denoising_steps - 1, (1,)).item()
            
            with torch.no_grad():
                seed = random.randint(0, 2*15)
                denoised_latents_low, low_noise = train_util.get_noisy_latent(
                    lat1, 
                    torch.manual_seed(seed), 
                    prior, 
                    noise_scheduler, 
                    timesteps_to, 
                    dtype, 
                    num_embeddings, 
                    embedding_dim
                    )
                denoised_latents_high, high_noise = train_util.get_noisy_latent(
                    lat2, 
                    torch.manual_seed(seed), 
                    prior, 
                    noise_scheduler, 
                    timesteps_to, 
                    dtype, 
                    num_embeddings, 
                    embedding_dim
                    )
                
            noise_scheduler.set_timesteps(1000)
            current_timestep = noise_scheduler.timesteps[int(timesteps_to * 1000 / args.max_denoising_steps)]
            optimizer.zero_grad()
            
            network.set_lora_slider(scale=scale_to_look)
            high_text_embeddings = empty_text_embedding(tokenizer, text_encoder, device, args.batch_size)
            with network:
                target_latents_high = train_util.predict_noise_shape(
                    prior, 
                    noise_scheduler, 
                    current_timestep, 
                    denoised_latents_high, 
                    high_text_embeddings, 
                    args.guidance_scale
                    ).to("cpu", dtype=torch.float32)
            loss_high = criteria(target_latents_high, high_noise.cpu().to(torch.float32))
            loss_high.backward()
            loss_high_for_epoch += loss_high.item()
            
            low_text_embeddings = empty_text_embedding(tokenizer, text_encoder, device, args.batch_size)
            network.set_lora_slider(scale=-scale_to_look)
            with network:
                target_latents_low = train_util.predict_noise_shape(
                prior, 
                noise_scheduler, 
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
        del (target_latents_low, target_latents_high, low_text_embeddings, high_text_embeddings)
        flush()
        
        if i % args.test_steps == 0:
            for scale in [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]:
                noise_scheduler.set_timesteps(50, device)
                latents = test_noise * noise_scheduler.init_noise_sigma
                latents = latents.reshape(latents.shape[0], num_embeddings, embedding_dim)
                latents = latents.to(dtype)
                network.set_lora_slider(scale)
                for t in tqdm(noise_scheduler.timesteps, desc=f"scale_{scale}"):
                    with network:
                        with torch.no_grad():
                            guided_target = train_util.predict_noise_shape(
                                prior, 
                                noise_scheduler, 
                                t, 
                                latents, 
                                test_text_embeddings, 
                                args.guidance_scale
                                )
                    latents = noise_scheduler.step(guided_target, t, latents).prev_sample
                image = shap_e_renderer.decode_to_image(latents[0, None, :], device, size=256).cpu().numpy()[11]   
                log_data[f"scale_{scale}"] = [wandb.Image(image.astype('uint8'), caption=f"scale_{scale}")]
        
        wandb.log(log_data)
        
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(save_path / "last.pt", dtype=dtype)

if __name__ == "__main__":
    args = config_util.parse_args()
    save_path = config_util.parse_save_path(args)
    print("save_path: ", save_path)
    folders, scales = config_util.parse_folders_and_scales(args)
    assert folders.shape[0] == scales.shape[0], "The number of folders and scales must be the same."
    print("args: ", args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    train(args, device, folders, scales, save_path)
