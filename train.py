import os
from typing import Optional, List, Optional
import json

import torch
import torch.utils.data
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EulerDiscreteScheduler,
    UNet2DConditionModel,
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
from video_diffusion.pipelines.stable_diffusion import SpatioTemporalStableDiffusionPipeline


logger = get_logger(__name__)

gif_duration = 200


def collate_fn(examples):
    batch = {
        "prompt_ids": torch.cat([example["prompt_ids"] for example in examples], dim=0),
        "images": torch.stack([example["images"] for example in examples]),
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
    for idx, prompt in enumerate(
        tqdm(
            prompts,
            desc="Generating sample images",
            disable=not accelerator.is_local_main_process,
        )
    ):
        generator = torch.Generator(device=accelerator.device)
        generator.manual_seed(idx)
        sequence = pipeline(
            prompt,
            num_inference_steps=20,
            generator=generator,
            clip_length=clip_length,
            guidance_scale=7,
        ).images[0]

        sequence = [annotate_text(image, prompt, font_size=15) for image in sequence]
        sequences.append(sequence)

    sequences = [make_grid(images, cols=2) for images in zip(*sequences)]
    sequences[0].save(
        image_save_path,
        save_all=True,
        append_images=sequences[1:],
        optimize=False,
        loop=0,
        duration=gif_duration,
    )


def train(
    pretrained_model_name_or_path: str,
    logdir: str,
    train_data_path: str,
    prompt: str,
    validation_steps: int = 1000,
    clip_length=8,
    sample_prompts: List[str] = None,
    train_steps: int = int(1e5),
    gradient_accumulation_steps: int = 1,
    seed: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    train_batch_size: int = 1,
    learning_rate: float = 5e-6,
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
    prior_preservation: Optional[float] = None,
    train_temporal_conv: bool = False,
    checkpointing_steps: int = 1000,
):
    args = get_function_args()

    time_string = get_time_string()
    logdir += f"_{time_string}"

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
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

    unet = UNetPseudo3DConditionModel.from_2d_model(
        os.path.join(pretrained_model_name_or_path, "unet"),
    )
    # unet = UNetPseudo3DConditionModel.from_pretrained(
    #     pretrained_model_name_or_path, 
    #     subfolder="unet",
    # )

    pipeline = SpatioTemporalStableDiffusionPipeline(
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

    if prior_preservation is not None:
        unet2d = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
        )

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            if prior_preservation is not None:
                unet2d.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if prior_preservation is not None:
        unet2d.requires_grad_(False)

    trainable_modules = ("attn_temporal", ".to_q")
    if train_temporal_conv:
        trainable_modules += "conv_temporal"
    for name, module in unet.named_modules():
        if name.endswith(trainable_modules):
            for params in module.parameters():
                params.requires_grad = True

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
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

    params_to_optimize = unet.parameters()
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

    # train_datasets = [
    #     ImageSequenceDataset(
    #         path=os.path.join(train_data_path, clip_name),
    #         n_sample_frame=clip_length,
    #         sampling_rate=10,
    #         stride=1,
    #         tokenizer=tokenizer,
    #         prompt=prompt,
    #     )
    #     for clip_name in os.listdir(train_data_path)
    # ]
    # train_dataset = torch.utils.data.ConcatDataset(train_datasets)

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
        collate_fn=collate_fn,
    )

    train_sample_save_path = os.path.join(logdir, "train_samples.gif")
    train_samples = []
    for idx, batch in enumerate(train_dataloader):
        if idx >= 4:
            break
        train_samples.append(batch["images"])
    train_samples = torch.cat(train_samples)
    train_samples = (train_samples.numpy() * 0.5 + 0.5).clip(0, 1)
    train_samples = rearrange(train_samples, "b c f h w -> b f h w c")
    train_samples = SpatioTemporalStableDiffusionPipeline.numpy_to_pil(train_samples)
    train_samples = [make_grid(images, cols=2) for images in zip(*train_samples)]
    train_samples[0].save(
        train_sample_save_path,
        save_all=True,
        append_images=train_samples[1:],
        optimize=False,
        loop=0,
        duration=gif_duration,
    )

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=train_steps * gradient_accumulation_steps,
    )

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
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    if prior_preservation is not None:
        unet2d.to(accelerator.device, dtype=weight_dtype)

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
    step = 0

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
        range(step, train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    def make_data_yielder(dataloader):
        while True:
            for batch in dataloader:
                yield batch
            accelerator.wait_for_everyone()

    train_data_yielder = make_data_yielder(train_dataloader)

    while step < train_steps:
        batch = next(train_data_yielder)

        vae.eval()
        text_encoder.eval()
        unet.train()
        if prior_preservation is not None:
            unet2d.eval()

        # with accelerator.accumulate(unet):
        # Convert images to latent space
        images = batch["images"].to(dtype=weight_dtype)
        b = images.shape[0]
        images = rearrange(images, "b c f h w -> (b f) c h w")
        latents = vae.encode(images).latent_dist.sample()
        latents = rearrange(latents, "(b f) c h w -> b c f h w", b=b)
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

        if prior_preservation is not None:
            model_pred_2d = unet2d(noisy_latents[:, :, 0], timesteps, encoder_hidden_states).sample
            loss = (
                loss
                + F.mse_loss(model_pred[:, :, 0].float(), model_pred_2d.float(), reduction="mean")
                * prior_preservation
            )

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)
            step += 1

            if sample_prompts and step % validation_steps == 0:
                unet.eval()
                if accelerator.is_main_process:
                    save_sample_images(
                        accelerator=accelerator,
                        prompts=sample_prompts,
                        pipeline=pipeline,
                        clip_length=clip_length,
                        step=step,
                        logdir=logdir,
                    )

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=step)

    accelerator.end_training()
    print("TRAINING DONE!")


if __name__ == "__main__":
    # args = dict(
    #     pretrained_model_name_or_path="/data/ckpt/stable-diffusion-v1-5-3d",
    #     pretrained_2d_model_name_or_path="/data/ckpt/stable-diffusion-v1-5",
    #     logdir="/data/tune-a-video/ckpt/debug",
    #     train_data_path="/data/data/video/sample/frames/02",
    #     prompt="a model is posing",
    #     validation_steps=100,
    #     checkpointing_steps=50000,
    #     clip_length=8,
    #     sample_prompts=[
    #         "a female model with long blonde hair is posing",
    #         "a male model with short black hair is posing",
    #         "a cat is posing",
    #         "a cat",
    #     ],
    #     seed=0,
    #     learning_rate=3e-5,
    # )

    args = dict(
        # pretrained_model_name_or_path="/data/ckpt/stable-diffusion-v1-5-3d",
        pretrained_model_name_or_path="/data/ckpt/stable-diffusion-v1-5",
        logdir="/data/tune-a-video/ckpt/debug",
        train_data_path="/data/data/video/sample/surfing/02",
        prompt="a man is surfing a wave",
        validation_steps=100,
        checkpointing_steps=50000,
        clip_length=8,
        sample_prompts=[
            "a man is surfing a wave",
            "a lady  in a yellow dress is surfing a wave",
            "a sloth is surfing a wave",
            "a man is surfing in a desert",
        ],
        seed=0,
        learning_rate=3e-5,
        train_steps=1000,
        prior_preservation=None,
        train_temporal_conv=False,
    )
    train(**args)
