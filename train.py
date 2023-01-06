import itertools
import os
import inspect
from typing import Optional, List
import json

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    EulerDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from einops import rearrange

from video_diffusion.models.unet_3d_condition import UNetPseudo3DConditionModel
from video_diffusion.data.dataset import ImageSequenceDataset
from video_diffusion.common.util import get_time_string, get_function_args
from video_diffusion.common.image_util import make_grid, annotate_text
from video_diffusion.pipelines.pipeline_stable_diffusion import StableDiffusionPipeline

from lora_diffusion import (
    inject_trainable_lora,
    save_lora_weight,
    monkeypatch_lora,
    tune_lora_scale,
)


logger = get_logger(__name__)

gif_duration = 200


def collate_fn(examples):
    batch = {
        "prompt_ids":  torch.cat([example["prompt_ids"] for example in examples], dim=0),
        "images": torch.stack([example["images"] for example in examples])
    }
    return batch


def save_sample_images(
    accelerator,
    prompts,
    pipeline,
    step,
    logdir,
    clip_length,
):
    image_save_root = os.path.join(logdir, "sample")
    os.makedirs(image_save_root, exist_ok=True)
    image_save_path = os.path.join(image_save_root, f"step_{str(step).zfill(6)}.gif")

    sequences = []
    for idx, prompt in enumerate(tqdm(
        prompts,
        desc="Generating sample images",
        disable=not accelerator.is_local_main_process,
    )):
        generator = torch.Generator(device=accelerator.device)
        generator.manual_seed(idx)
        sequence = pipeline(
            prompt,
            num_inference_steps=20,
            generator=generator,
            frames_length=clip_length,
            guidance_scale=7,
        ).images[0]

        sequence = [annotate_text(image, prompt, font_size=15) for image in sequence]
        sequences.append(sequence)

    sequences = [make_grid(images, cols=2) for images in zip(*sequences)]
    sequences[0].save(image_save_path, save_all=True, append_images=sequences[1:], optimize=False, loop=0, duration=gif_duration)


def train(
    pretrained_model_name_or_path: str,
    logdir: str,
    train_data_path: str,
    prompt: str,
    validation_steps: int = 1000,
    clip_length=8,
    sample_prompts: List[str] = None,
    train_steps: int = int(1e5),
    train_text_encoder: bool = False,
    gradient_accumulation_steps: int = 1,
    seed: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    train_batch_size: int = 1,
    lora: bool = False,
    lora_rank: int = 4,
    learning_rate: float = 5e-6,
    learning_rate_text_encoder: Optional[float] = None,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",  # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    lr_warmup_steps: int = 0,
    use_8bit_adam: bool = True,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_checkpointing: bool = False,
    checkpointing_steps: int = 1000,
):
    # TODO: inspect signature & save args
    args = get_function_args()

    time_string = get_time_string()
    logdir += f"_{time_string}"

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if (
        train_text_encoder
        and gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )


    if seed is not None:
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, "config.json"), "w") as f:
            json.dump(args, f, indent=2)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
    )

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
    )

    unet = UNetPseudo3DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
    )

    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=EulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
        ),
    )
    pipeline.set_progress_bar_config(disable=True)

    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if lora:
        unet_lora_params, _ = inject_trainable_lora(unet, r=lora_rank)
        if train_text_encoder:
            text_encoder_lora_params, _ = inject_trainable_lora(
                text_encoder,
                target_replace_module=["CLIPAttention"],
                r=lora_rank,
            )
    else:
        if train_text_encoder:
            text_encoder.requires_grad_(True)

        for name, module in unet.named_modules():
            if "temporal" in name:
                for params in module.parameters():
                    params.requires_grad = True


    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if scale_lr:
        learning_rate = (
            learning_rate
            * gradient_accumulation_steps
            * train_batch_size
            * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    text_lr = learning_rate if learning_rate_text_encoder is None else learning_rate_text_encoder

    if lora:
        params_to_optimize = (
            [
                {"params": itertools.chain(*unet_lora_params), "lr": learning_rate},
                {
                    "params": itertools.chain(*text_encoder_lora_params),
                    "lr": text_lr,
                },
            ]
            if train_text_encoder
            else itertools.chain(*unet_lora_params)
        )
    else:
        params_to_optimize = (
            itertools.chain(unet.parameters(), text_encoder.parameters())
            if train_text_encoder
            else unet.parameters()
        )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    train_dataset = ImageSequenceDataset(
        path=train_data_path,
        n_sample_frame=clip_length,
        sampling_rate=10,
        stride=1,
        tokenizer=tokenizer,
        prompt=prompt,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    train_sample_save_path = os.path.join(logdir, "train_samples.gif")
    train_samples = []
    for idx, batch in enumerate(train_dataloader):
        if idx >= 4:
            break
        train_samples.append(batch["images"])
    train_samples = torch.cat(train_samples)
    train_samples = (train_samples.numpy( ) * 0.5 + 0.5).clip(0, 1)
    train_samples = rearrange(train_samples, "b c f h w -> b f h w c")
    train_samples = StableDiffusionPipeline.numpy_to_pil(train_samples)
    train_samples = [make_grid(images, cols=2) for images in zip(*train_samples)]
    train_samples[0].save(train_sample_save_path, save_all=True, append_images=train_samples[1:], optimize=False, loop=0, duration=gif_duration)

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=train_steps * gradient_accumulation_steps,
    )

    if train_text_encoder:
        (
            unet,
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(unet, text_encoder, optimizer, train_dataloader, lr_scheduler)
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )
    accelerator.register_for_checkpointing(lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("video")  # , config=vars(args))

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Instantaneous batch size per device = {train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    print(f"  Total optimization steps = {train_steps}")
    global_step = 0
    epoch = 0

    if sample_prompts:
        save_sample_images(
            accelerator=accelerator,
            pipeline=pipeline,
            prompts=sample_prompts,
            clip_length=clip_length,
            step=0,
            logdir=logdir,
        )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    while global_step < train_steps:
        for step, batch in enumerate(train_dataloader):
            vae.eval()
            unet.train()
            if train_text_encoder:
                text_encoder.train()

            # with accelerator.accumulate(unet):
            # Convert images to latent space
            images = batch["images"].to(dtype=weight_dtype)
            b = images.shape[0]
            images  = rearrange(images, "b c f h w -> (b f) c h w")
            latents = vae.encode(images).latent_dist.sample()
            latents  = rearrange(latents, "(b f) c h w -> b c f h w", b=b)
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                params_to_clip = (
                    itertools.chain(unet.parameters(), text_encoder.parameters())
                    if train_text_encoder
                    else unet.parameters()
                )
                accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                unet.eval()
                if train_text_encoder:
                    text_encoder.eval()

                if sample_prompts and global_step % validation_steps == 0:
                    if accelerator.is_main_process:
                        save_sample_images(
                            accelerator=accelerator,
                            prompts=sample_prompts,
                            pipeline=pipeline,
                            clip_length=clip_length,
                            step=global_step,
                            logdir=logdir,
                        )

                if global_step % checkpointing_steps == 0:

                    if global_step == 10000:
                        checkpointing_steps = checkpointing_steps * 10

                    if accelerator.is_main_process:
                        # newer versions of accelerate allow the 'keep_fp32_wrapper' arg. without passing
                        # it, the models will be unwrapped, and when they are then used for further training,
                        # we will crash. pass this, but only to newer versions of accelerate. fixes
                        # https://github.com/huggingface/diffusers/issues/1566
                        accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                            inspect.signature(accelerator.unwrap_model).parameters.keys()
                        )
                        extra_args = {"keep_fp32_wrapper": True} if accepts_keep_fp32_wrapper else {}
                        pipeline = DiffusionPipeline.from_pretrained(
                            pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet, **extra_args),
                            text_encoder=accelerator.unwrap_model(text_encoder, **extra_args),
                        )

                        if lora:
                            filename_unet = f"{logdir}/lora_weight_e{epoch}_s{global_step}.pt"
                            filename_text_encoder = (
                                f"{logdir}/lora_weight_e{epoch}_s{global_step}.text_encoder.pt"
                            )
                            print(f"save weights {filename_unet}, {filename_text_encoder}")
                            save_lora_weight(pipeline.unet, filename_unet)
                            if train_text_encoder:
                                save_lora_weight(
                                    pipeline.text_encoder,
                                    filename_text_encoder,
                                    target_replace_module=["CLIPAttention"],
                                )
                        else:
                            save_path = os.path.join(logdir, f"checkpoint-{global_step}")
                            pipeline.save_pretrained(save_path)
                            print(f"Saved state to {save_path}")
                            filename_unet = None
                            filename_text_encoder = None

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= train_steps:
                break

        accelerator.wait_for_everyone()

        epoch += 1


    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        pipeline = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
        )
        if lora:
            save_lora_weight(pipeline.unet, logdir + "/lora_weight.pt")
            if train_text_encoder:
                save_lora_weight(
                    pipeline.text_encoder,
                    logdir + "/lora_weight.text_encoder.pt",
                    target_replace_module=["CLIPAttention"],
                )
        else:
            pipeline.save_pretrained(logdir)

    accelerator.end_training()
    print("TRAINING DONE!")


if __name__ == "__main__":
    args = dict(
        pretrained_model_name_or_path="/data/ckpt/stable-diffusion-v1-5-3d",
        logdir="/data/tune-a-video/ckpt/debug",
        train_data_path="/data/data/video/sample/00/frames",
        prompt="a video of a model photoshoot",
        validation_steps=500,
        checkpointing_steps=50000,
        clip_length=8,
        sample_prompts=[
            "a video of a model photoshoot",
            "a model photoshoot",
            "a photoshoot",
            "a model",
        ],
        seed=0,
    )
    train(**args)
