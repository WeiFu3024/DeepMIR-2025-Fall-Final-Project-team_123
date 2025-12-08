print("Warning: This training script is specifically designed for training MuseControlLite with audio-only inputs (drum + vocal conditions). For training, please move the file to MuseControlLite repository root directory and run it there.")
import argparse
import itertools
import math
import os
import random
import shutil
import warnings
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import Dataset, random_split, DataLoader
import torchaudio
from tqdm.auto import tqdm
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from scipy.io.wavfile import write
from scipy.signal import savgol_filter
from diffusers.loaders import AttnProcsLayers
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file  # Import safetensors
warnings.filterwarnings("ignore", category=FutureWarning)
from utils.stable_audio_dataset_utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T
from torchaudio import transforms as T
import torch
import soundfile as sf
from pipeline.stable_audio_multi_cfg_pipe import StableAudioPipeline
from diffusers.loaders import AttnProcsLayers
from MuseControlLite_attn_processor import (
    StableAudioAttnProcessor2_0,
    StableAudioAttnProcessor2_0_rotary,
    StableAudioAttnProcessor2_0_rotary_double,
    # StableAudioAttnProcessor2_0_rotary_no_v,
)
from utils.extract_conditions import compute_dynamics, extract_melody_one_hot, evaluate_f1_rhythm
from config_training import get_config
import gc
try:
    torchaudio.set_audio_backend("sox_io")
except AttributeError:
    pass
import torch.distributed as dist
from safetensors.torch import save_file
from torch.cuda.amp import autocast

def save_attn_processors(pipeline, out_dir, filename="attn_procs.safetensors"):
    os.makedirs(out_dir, exist_ok=True)
    state = {}
    for name, proc in pipeline.transformer.attn_processors.items():
        # grab all registered parameters on the processor
        for p_name, p in proc.named_parameters(recurse=True):
            state[f"{name}.{p_name}"] = p.detach().cpu()
    # nothing to save if processors are stateless
    if len(state) == 0:
        print("No trainable parameters found in attn processors (they might be stateless).")
    save_file(state, os.path.join(out_dir, filename))
    print(f"Saved {len(state)} tensors to {os.path.join(out_dir, filename)}")
class AudioInversionDataset(Dataset):
    def __init__(
        self,
        config,
        audio_codec_root, # drum latent
        vocal_codec_root, # vocal latent
        audio_data_root, 
        device,
        force_channels="stereo"
    ):
        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
        )
        self.root_paths = []
        self.force_channels = force_channels
        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )
        self.config = config
        self.audio_data_root = audio_data_root
        self.vocal_codec_root = vocal_codec_root
        self.audio_codec_root = audio_codec_root
        self.device = device
        self.meta_path = config['meta_data_path']
        with open(self.meta_path) as f:
            self.meta = json.load(f)
    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):    
        meta_entry = self.meta[i]
        audio_full_path = meta_entry.get('path')
        filename = os.path.basename(audio_full_path)
        
        # Load audio tokens, they are encoded with the Stable-audio VAE and saved, skipping the the VAE encoding process saves memory when training MuseControlLite
        # audio_full_path = os.path.join(self.audio_data_root, filename)
        
        drum_token_path = os.path.join(self.audio_codec_root, filename.replace('wav', 'pt'))
        vocal_token_path = os.path.join(self.vocal_codec_root, filename.replace('wav', 'pt'))
        
        # debug checking
        if not os.path.exists(drum_token_path):
            raise FileNotFoundError(f"Drum token file not found: {drum_token_path}")
        if not os.path.exists(vocal_token_path):
            raise FileNotFoundError(f"Vocal token file not found: {vocal_token_path}")
        
        # audio_token_path = os.path.join(self.audio_codec_root, audio_path.replace('mp3', 'pth'))
        # audio = torch.load(audio_token_path, map_location=torch.device('cpu'))
        
        drum_latent = torch.load(drum_token_path, map_location='cpu')
        if drum_latent.dim() == 3: drum_latent = drum_latent.squeeze(0)

        vocal_latent = torch.load(vocal_token_path, map_location='cpu')
        if vocal_latent.dim() == 3: vocal_latent = vocal_latent.squeeze(0)
        
        MAX_LATENT_LEN = 1024
        
        # Pad to MAX_LATENT_LEN
        
        def pad_tensor(t, target_len):
            curr_len = t.shape[-1]
            if curr_len < target_len:
                return F.pad(t, (0, target_len - curr_len), "constant", 0)
            else:
                return t[:, :target_len]

        drum_latent = pad_tensor(drum_latent, MAX_LATENT_LEN)
        vocal_latent = pad_tensor(vocal_latent, MAX_LATENT_LEN)
        
        original_seconds = meta_entry.get('duration', 2097152 / 44100)
        
        example = {
            "text": "a hip-pop drum beat",  
            "audio_full_path": audio_full_path,
            "audio": drum_latent,
            "vocal_audio": vocal_latent,
            "seconds_start": 0,
            "seconds_end": original_seconds,
        }
        return example
    
class CollateFunction:
    def __init__(self, condition_type):
        self.condition_type = condition_type
    def __call__(self, examples):
        audio = [example["audio"] for example in examples] # drum
        vocal_audio = [example["vocal_audio"] for example in examples] # vocal
        
        prompt_texts = [example["text"] for example in examples]
        audio_full_path = [example["audio_full_path"] for example in examples]
        seconds_start = [example["seconds_start"] for example in examples]
        seconds_end = [example["seconds_end"] for example in examples]
        
        audio = torch.stack(audio)
        vocal_audio = torch.stack(vocal_audio)
        
        batch = {
            "audio_full_path": audio_full_path,
            "audio": audio,                 # [B, 64, T]
            "vocal_audio": vocal_audio,     # [B, 64, T]
            "prompt_texts": prompt_texts,
            "seconds_start": seconds_start,
            "seconds_end": seconds_end,
        }

        return batch

def log_validation(val_dataloader, condition_type, pipeline, config, weight_dtype, global_step):
    val_audio_dir = os.path.join(config["output_dir"], "val_audio_{}".format(global_step))
    os.makedirs(val_audio_dir, exist_ok=True)
    # score_dynamics = []
    # score_melody = []
    # score_rhythm = []
    
    for step, batch in enumerate(val_dataloader):
        if step > config["test_num"]:
            break
        pipeline.transformer.eval()  # Set the transformer to evaluation mode
        
        prompt_texts = batch["prompt_texts"]
        audio_full_path = batch["audio_full_path"]
        vocal_condition = batch["vocal_audio"].to(pipeline.device)
        
        ### conditioned
        # audio_condition = batch["audio"]
        desired_repeats = 768 // 64  # Number of repeats needed
        extracted_audio_condition = vocal_condition.repeat_interleave(desired_repeats, dim=1)
        
        empty_condition = torch.zeros_like(extracted_audio_condition)
        extracted_condition = torch.concat([empty_condition, empty_condition, extracted_audio_condition], dim=0)
        
        # masked_extracted_audio_condition = torch.full_like(extracted_audio_condition.to(torch.float32), fill_value=0)
        # extracted_audio_condition[:,:,512:] = 0
        # extracted_condition = torch.concat((masked_extracted_audio_condition, masked_extracted_audio_condition, extracted_audio_condition), dim=0)
        # extracted_condition = extracted_condition.transpose(1, 2)
        
        extracted_condition = extracted_condition.transpose(1, 2)
        generator = torch.Generator("cuda").manual_seed(0)
        # print(batch["seconds_end"][0])
        with torch.no_grad():
            audio = pipeline(
                extracted_condition = extracted_condition, 
                guidance_scale_con = config['guidance_scale_con'],
                guidance_scale_text=config["guidance_scale_text"],
                prompt=prompt_texts,
                negative_prompt=[""],
                num_inference_steps=config["denoise_step"],
                audio_end_in_s=batch["seconds_end"][0],
                num_waveforms_per_prompt=1,
                generator=generator,
            ).audios
            
        output = audio[0].T.float().cpu().numpy()
        gen_file = os.path.join(val_audio_dir, f"validation_{step}.wav")
        original_file = os.path.join(val_audio_dir, f"original_{step}.wav")
        sf.write(gen_file, output, pipeline.vae.sampling_rate)
        shutil.copy(audio_full_path[0], original_file)
        discription_path = os.path.join(val_audio_dir, "description.txt")
        with open(discription_path, 'a') as file:
            file.write(f'{prompt_texts}\n')
            
    torch.cuda.empty_cache()
    gc.collect()
    return 0, 0, 0

def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)
def check_and_print_non_float32_parameters(model):
    non_float32_params = []
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            non_float32_params.append((name, param.dtype))
    
    if non_float32_params:
        print("Not all parameters are in float32!")
        print("The following parameters are not in float32:")
        for name, dtype in non_float32_params:
            print(f"Parameter: {name}, Data Type: {dtype}")
    else:
        print("All parameters are in float32.")
import torch
import math

# def _insert_peak_(weights: torch.Tensor,
#                   center: int,
#                   width: int = 10,
#                   peak: float = 2.0):
#     """
#     In-place 將 raised-cosine 權重插入 weights 末軸 (time 軸)。
#     將 curve 的 dtype / device 與 weights 一致，避免 RuntimeError。
#     """
#     T = weights.shape[-1]
#     left  = max(center - width, 0)
#     right = min(center + width + 1, T)

#     idx   = torch.arange(left, right, device=weights.device)        # int64
#     dist  = (idx - center).abs().to(weights.dtype)                  # ⇦ dtype 對齊

#     # raised-cosine 0~1
#     smooth = 0.5 * (1 + torch.cos(math.pi * dist / width))
#     curve  = 1.0 + (peak - 1.0) * smooth                            # float16 / float32…
#     # 若前一步已經轉 dtype，可省略下行；保險起見再 to 一次
#     curve  = curve.to(dtype=weights.dtype, device=weights.device)
#     # in-place 取最大值
#     weights[..., idx] = torch.maximum(weights[..., idx], curve)
#     return weights


def main():
    torch.manual_seed(42)
    config = get_config()
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'
    os.environ['CUDA_VISIBLE_DEVICES'] = config["GPU_id"]
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision=config["mixed_precision"],
        log_with="wandb",
    )

    if not is_wandb_available():
        raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    
    # Handle the repository creation
    if accelerator.is_main_process:
        if config["output_dir"] is not None:
            os.makedirs(config["output_dir"], exist_ok=True)
    # decide weight precision for freezed models
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # initialize models
    pipeline = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=weight_dtype).to("cuda")
    text_encoder=pipeline.text_encoder
    projection_model=pipeline.projection_model
    vae=pipeline.vae
    noise_scheduler=pipeline.scheduler
    noise_scheduler.config.sigma_max = config["sigma_max"]
    noise_scheduler.config.sigma_min = config["sigma_min"]
    transformer = pipeline.transformer

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    projection_model.requires_grad_(False)

    # Define a dictionary to map types to corresponding processor classes, currently only "rotary" is available.
    processor_classes = {
        "rotary": StableAudioAttnProcessor2_0_rotary,
        "rotary_double": StableAudioAttnProcessor2_0_rotary_double,
        # "no_v": StableAudioAttnProcessor2_0_rotary_no_v,
    }
    print(config["attn_processor_type"])
    # Get the processor classes based on the type
    attn_processor = processor_classes.get(config["attn_processor_type"], None)
    attn_procs = {}
    for name in transformer.attn_processors.keys():
        if name.endswith("attn1.processor"):
            attn_procs[name] = StableAudioAttnProcessor2_0()
        else:
            attn_procs[name] = attn_processor(
                layer_id = name.split(".")[1],
                hidden_size=768,
                name=name,
                cross_attention_dim=768,
                scale=config['ap_scale'],
            ).to("cuda", dtype=torch.float32)
    # Load checkpoint
    if config["transformer_ckpt"] is not None:
        if "bin" in config["transformer_ckpt"]:
            state_dict = torch.load(config["transformer_ckpt"])
        elif "safetensors" in config["transformer_ckpt"]:
            state_dict = load_file(config["transformer_ckpt"], device="cpu")
        for name, processor in attn_procs.items():
            if isinstance(processor, attn_processor):
                weight_name_v = name + ".to_v_ip.weight"
                weight_name_k = name + ".to_k_ip.weight"
                conv_out_weight = name + ".conv_out.weight"
                processor.to_v_ip.weight = torch.nn.Parameter(state_dict[weight_name_v].to(torch.float32))
                processor.to_k_ip.weight = torch.nn.Parameter(state_dict[weight_name_k].to(torch.float32))
                processor.conv_out.weight = torch.nn.Parameter(state_dict[conv_out_weight].to(torch.float32))
                print(f"load {name}")
    transformer.set_attn_processor(attn_procs)
    # class _Wrapper(AttnProcsLayers):
    #     def forward(self, *args, **kwargs):
    #         return pipeline.transformer(*args, **kwargs)

    # transformer = _Wrapper(pipeline.transformer.attn_processors)
    proc_container = AttnProcsLayers(pipeline.transformer.attn_processors)
    qkv_params = []
    core = pipeline.transformer                                            
    from torch.cuda.amp import autocast

    for mod_name, m in pipeline.transformer.named_modules():
        if ".attn1" in mod_name and all(hasattr(m, n) for n in ("to_q", "to_k", "to_v")):
            for lin_name in ("to_q", "to_k", "to_v"):
                lin = getattr(m, lin_name)

                # keep params in FP32
                lin.to(dtype=torch.float32)

                # correct wrapper: use the unbound class method and pass self explicitly
                orig_fwd = type(lin).forward
                def _fp32_forward(self, x, *args, **kwargs):
                    with autocast(enabled=False):
                        return orig_fwd(self, x.float(), *args, **kwargs)

                lin.forward = _fp32_forward.__get__(lin, type(lin))

                # make sure trainable
                for p in lin.parameters():
                    p.requires_grad_(True)
                qkv_params += list(lin.parameters())

    optimizer_class = torch.optim.AdamW
    params_to_optimize = itertools.chain(
        proc_container.parameters(), qkv_params
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay= config['weight_decay'],
        eps=1e-08,
    )

    # Dataset and DataLoaders creation:
    dataset = AudioInversionDataset(
        config,
        audio_codec_root=config['audio_codec_root'],
        vocal_codec_root=config['vocal_codec_root'],
        audio_data_root=config["audio_data_dir"],
        device=accelerator.device,
        )
    val_size =  config["validation_num"]
    train_size = len(dataset) - val_size 

    # Ensure consistent splitting
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # DataLoader
    train_collate_fn = CollateFunction(condition_type=config["condition_type"])
    val_collate_fn = CollateFunction(condition_type=config["condition_type"])
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=config["dataloader_num_workers"],
        pin_memory=True,
        prefetch_factor=1,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=val_collate_fn,
        num_workers=config["dataloader_num_workers"],
        pin_memory=True,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"])
    if config["max_train_steps"] is None:
        config["max_train_steps"] = config["num_train_epochs"] * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        config['lr_scheduler'],
        optimizer=optimizer,
        step_rules = None,
        num_warmup_steps = 500,
        num_training_steps = config['max_train_steps'],
        num_cycles = 1,
        power = 1.0,
        last_epoch = -1,
    )
    print("Before prepare:", core is pipeline.transformer)  # True
    # Prepare everything with our `accelerator`.
    core, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        core, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    print("After prepare:", core is pipeline.transformer)   # False
    underlying = accelerator.unwrap_model(core)
    print("Underlying is original:", underlying is pipeline.transformer)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config["gradient_accumulation_steps"])
    if overrode_max_train_steps:
        config["max_train_steps"] = config["num_train_epochs"] * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    config["num_train_epochs"] = math.ceil(config["max_train_steps"] / num_update_steps_per_epoch)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="Rap2Beat",      # your W&B project
            config=config,                        # whatever hyperparams you’re logging
            init_kwargs={
                "wandb": {
                    "name": config['wand_run_name'],   # <— your chosen run name
                }
            }
        )
    global_step = 0
    first_epoch = 0
    score_melody = 0
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, config["max_train_steps"]), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    print("log_validation_first", config["log_first"])
    score_dynamics, score_melody, score_rhythm = 0, 0, 0
    if config["log_first"] and accelerator.is_main_process:
        score_dynamics, score_melody, score_rhythm = log_validation(val_dataloader,
                            config["condition_type"],
                            pipeline, config, weight_dtype, global_step
                        )
    for epoch in range(first_epoch, config["num_train_epochs"]):
        for step, batch in enumerate(train_dataloader):
            core.train()
            with accelerator.accumulate(core):
                # Convert audios to latent space
                latents = batch["audio"]
                bsz, channels, height = latents.shape
                # Sample a random timestep for each image using uniform distribution
                t = torch.sigmoid(torch.randn(bsz)).cuda().half()
                # Calculate the noise schedule parameters for those timesteps
                alphas, sigmas = get_alphas_sigmas(t)  # get_alphas_sigmas should be defined as in the wrapper
                alphas = alphas[:, None, None]  # Shape to match latents
                sigmas = sigmas[:, None, None]
                # Sample noise and add it to the latents
                noise = torch.randn_like(latents)
                noisy_latents = latents * alphas + noise * sigmas
                # Determine the target for v_prediction
                if noise_scheduler.config.prediction_type == "v_prediction":
                    targets = alphas * noise - sigmas * latents
                else:
                    targets = noise  # For epsilon, the target is just the noise
                    
                prompt_texts = batch["prompt_texts"]
                vocal_latents = batch["vocal_audio"]
                
                desired_repeats = 768 // 64  # Number of repeats needed
                extracted_audio_condition = vocal_latents.repeat_interleave(desired_repeats, dim=1)
                
                # Add dropout and random empty prompt
                for i in range(len(prompt_texts)):
                    rand_num = random.random()
                    
                    # 10% 機率丟掉 Text (變成無文字引導)
                    if random.random() < 0.1:
                        prompt_texts[i] = ""
                    
                    # 10% 機率丟掉 Vocal Condition (變成純粹隨機生成)
                    # 這能幫助模型學會 "無條件生成"，避免過度依賴
                    if rand_num < 0.1:
                        extracted_audio_condition[i] = 0
                
                # B, _, T = extracted_audio_condition.shape
                # w = torch.ones((B, 64, T), device="cuda")
                # for i in range(len(prompt_texts)):
                #     rand_num = random.random()
                #     num1, num2 = random.sample(range(1024), 2)
                #     # 50% chance to set prompt_texts[i] to an empty string
                #     if random.random() < 0.5:
                #         prompt_texts[i] = ""           
                #     if rand_num < 0.1:
                #          extracted_audio_condition[i] = 0
                #     else:
                #         segment_length = random.randint(340, 900)
                #         start_index = random.randint(
                #             0, 
                #             extracted_audio_condition[i].shape[1] - segment_length
                #         )
                #         extracted_audio_condition[i][:, start_index : start_index + segment_length] = 0
                        # w[i] = _insert_peak_(w[i], start_index,               width=200, peak=2.0)
                        # w[i] = _insert_peak_(w[i], start_index + segment_length - 1,width=200, peak=2.0)
                with torch.no_grad():
                    prompt_embeds = pipeline.encode_prompt(
                        prompt=prompt_texts,
                        device="cuda",
                        do_classifier_free_guidance=False,
                    )
                    batch_size = len(prompt_texts)
                    audio_start_in_s = batch["seconds_start"]
                    audio_end_in_s = batch["seconds_end"]
                    # Encode duration
                    seconds_start_hidden_states, seconds_end_hidden_states = pipeline.encode_duration(
                        audio_start_in_s,
                        audio_end_in_s,
                        device="cuda",
                        do_classifier_free_guidance=False,
                        batch_size=batch_size,
                    )
                
                audio_duration_embeds = torch.cat([seconds_start_hidden_states, seconds_end_hidden_states], dim=2)
                text_audio_duration_embeds = torch.cat(
                    [prompt_embeds, seconds_start_hidden_states, seconds_end_hidden_states], dim=1
                ) 
                extracted_condition = extracted_audio_condition.transpose(1, 2)
                # This rotary_embedding is for self attention layers in Stable-audio 
                rotary_embed_dim = pipeline.transformer.config.attention_head_dim // 2
                rotary_embedding = get_1d_rotary_pos_embed(
                    rotary_embed_dim,
                    latents.shape[2] + audio_duration_embeds.shape[1],
                    use_real=True,
                    repeat_interleave_real=False,
                )              
                with accelerator.autocast():
                    model_pred = core(
                        noisy_latents,
                        t,  # Use continuous t for conditioning
                        encoder_hidden_states=text_audio_duration_embeds,
                        encoder_hidden_states_con = extracted_condition,
                        global_hidden_states=audio_duration_embeds,
                        rotary_embedding=rotary_embedding,
                        return_dict=False,
                    )[0]
                    # Compute the loss
                    loss = F.mse_loss(model_pred, targets, reduction="mean")
                    # weighted_sq_err = sq_err * w 
                    # loss = weighted_sq_err.sum() / w.sum()   
                # Backpropagation
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = list(proc_container.parameters()) + list(qkv_params)
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    # gc.collect()
                    # torch.cuda.empty_cache()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                audios = []
                progress_bar.update(1)
                global_step += 1
            
                if accelerator.is_main_process:
                    if global_step % config["checkpointing_steps"] == 0:
                        from safetensors.torch import save_file  # pip install safetensors

                        save_dir = os.path.join(config["output_dir"], f"checkpoint-{global_step}")
                        os.makedirs(save_dir, exist_ok=True)

                        core_unwrap = accelerator.unwrap_model(core)
                        qkv_state = {
                            k: v.detach().cpu().contiguous()
                            for k, v in core_unwrap.state_dict().items()
                            if (".attn1.to_q." in k) or (".attn1.to_k." in k) or (".attn1.to_v." in k)
                        }

                        qkv_path = os.path.join(save_dir, f"attn1_qkv_{global_step}.safetensors")

                        # --- cleanup from previous bug: remove dir if it exists at the file path ---
                        if os.path.isdir(qkv_path):
                            import shutil
                            shutil.rmtree(qkv_path)

                        # save as .safetensors (optionally include metadata)
                        save_file(qkv_state, qkv_path, metadata={"global_step": str(global_step)})

                        accelerator.print(f"Saved QKV to {qkv_path}")
                        save_attn_processors(pipeline, save_dir)

                        # if other ranks will read anything under save_dir afterwards:
                        accelerator.wait_for_everyone()
                    if global_step % config["validation_steps"] == 0:
                        score_dynamics, score_melody, score_rhythm = log_validation(val_dataloader,
                            config["condition_type"],
                            pipeline, config, weight_dtype, global_step
                        )
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= config["max_train_steps"]:
                break
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()
    
