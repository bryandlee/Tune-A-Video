import os
import inspect
from typing import Optional, List, Dict

import click
from omegaconf import OmegaConf

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
    DDIMScheduler,
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
from video_diffusion.common.image_util import make_grid, annotate_image, save_images_as_gif
from video_diffusion.pipelines.stable_diffusion import SpatioTemporalStableDiffusionPipeline


logger = get_logger(__name__)


def collate_fn(examples):
    batch = {
        "prompt_ids": torch.cat([example["prompt_ids"] for example in examples], dim=0),
        "images": torch.stack([example["images"] for example in examples]),
    }
    return batch


def log_train_samples(
    train_dataloader,
    save_path,
    num_batch: int = 4,
):
    train_samples = []
    for idx, batch in enumerate(train_dataloader):
        if idx >= num_batch:
            break
        train_samples.append(batch["images"])

    train_samples = torch.cat(train_samples).numpy()
    train_samples = rearrange(train_samples, "b c f h w -> b f h w c")
    train_samples = (train_samples * 0.5 + 0.5).clip(0, 1)
    train_samples = SpatioTemporalStableDiffusionPipeline.numpy_to_pil(train_samples)
    train_samples = [make_grid(images, cols=2) for images in zip(*train_samples)]
    save_images_as_gif(train_samples, save_path)


class SampleLogger:
    def __init__(
        self,
        prompts: List[str],
        clip_length: int,
        logdir: str,
        subdir: str = "sample",
        num_samples_per_prompt: int = 2,
        num_inference_steps: int = 20,
        guidance_scale: float = 7,
        annotate: bool = True,
        annotate_size: int = 15,
    ) -> None:
        self.prompts = prompts
        self.clip_length = clip_length
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.num_samples_per_prompt = num_samples_per_prompt

        self.logdir = os.path.join(logdir, subdir)
        os.makedirs(self.logdir)

        self.annotate = annotate
        self.annotate_size = annotate_size

    def log_sample_images(
        self, pipeline: SpatioTemporalStableDiffusionPipeline, device: torch.device, step: int
    ):
        save_path = os.path.join(self.logdir, f"step_{step}.gif")
        image_sequences = []
        for prompt in tqdm(self.prompts, desc="Generating sample images"):
            for seed in range(self.num_samples_per_prompt):
                generator = torch.Generator(device=device)
                generator.manual_seed(seed)
                sequence = pipeline(
                    prompt,
                    generator=generator,
                    num_inference_steps=self.num_inference_steps,
                    clip_length=self.clip_length,
                    guidance_scale=self.guidance_scale,
                    num_images_per_prompt=1,
                ).images[0]

                if self.annotate:
                    images = [
                        annotate_image(image, prompt, font_size=self.annotate_size) for image in sequence
                    ]
                image_sequences.append(images)

        image_sequences = [make_grid(images, cols=2) for images in zip(*image_sequences)]
        save_images_as_gif(image_sequences, save_path)


def train(
    pretrained_model_path: str,
    logdir: str,
    train_dataset: Dict,
    train_steps: int = 300,
    validation_steps: int = 1000,
    validation_sample_logger: Optional[Dict] = None,
    gradient_accumulation_steps: int = 1,
    seed: Optional[int] = None,
    mixed_precision: Optional[str] = "fp16",
    train_batch_size: int = 1,
    learning_rate: float = 3e-5,
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
    if accelerator.is_main_process:
        os.makedirs(logdir, exist_ok=True)
        OmegaConf.save(args, os.path.join(logdir, "config.yml"))

    if seed is not None:
        set_seed(seed)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_path,
        subfolder="tokenizer",
        use_fast=False,
    )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path,
        subfolder="text_encoder",
    )

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_path,
        subfolder="vae",
    )

    unet = UNetPseudo3DConditionModel.from_2d_model(
        os.path.join(pretrained_model_path, "unet"),
    )

    pipeline = SpatioTemporalStableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=DDIMScheduler.from_pretrained(
            pretrained_model_path,
            subfolder="scheduler",
        ),
    )
    pipeline.set_progress_bar_config(disable=True)

    if prior_preservation is not None:
        unet2d = UNet2DConditionModel.from_pretrained(
            pretrained_model_path,
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
        pretrained_model_path,
        subfolder="scheduler",
    )

    prompt_ids = tokenizer(
        train_dataset["prompt"],
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids

    train_dataset = ImageSequenceDataset(**train_dataset, prompt_ids=prompt_ids)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
    )

    train_sample_save_path = os.path.join(logdir, "train_samples.gif")
    log_train_samples(save_path=train_sample_save_path, train_dataloader=train_dataloader)

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

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {train_steps}")
    step = 0

    if validation_sample_logger is not None and accelerator.is_main_process:
        validation_sample_logger = SampleLogger(**validation_sample_logger, logdir=logdir)
        validation_sample_logger.log_sample_images(
            pipeline=pipeline,
            device=accelerator.device,
            step=0,
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
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
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

            if accelerator.is_main_process:

                if validation_sample_logger is not None and step % validation_steps == 0:
                    unet.eval()
                    validation_sample_logger.log_sample_images(
                        pipeline=pipeline,
                        device=accelerator.device,
                        step=step,
                    )

                if step % checkpointing_steps == 0:
                    accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                        inspect.signature(accelerator.unwrap_model).parameters.keys()
                    )
                    extra_args = {"keep_fp32_wrapper": True} if accepts_keep_fp32_wrapper else {}
                    pipeline_save = SpatioTemporalStableDiffusionPipeline.from_pretrained(
                        pretrained_model_path,
                        unet=accelerator.unwrap_model(unet, **extra_args),
                    )
                    checkpoint_save_path = os.path.join(logdir, f"checkpoint_{step}")
                    pipeline_save.save_pretrained(checkpoint_save_path)

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=step)

    accelerator.end_training()


@click.command()
@click.option("--config", type=str, default="config/sample.yml")
def run(config):
    train(**OmegaConf.load(config))


if __name__ == "__main__":
    run()
