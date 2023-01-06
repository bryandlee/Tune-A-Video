import os

import numpy as np
from PIL import Image
from einops import rearrange

import torch
from torch.utils.data import Dataset

from .transform import short_size_scale, random_crop
from ..common.image_util import IMAGE_EXTENSION


class ImageSequenceDataset(Dataset):
    def __init__(
        self, 
        path: str,
        tokenizer,
        prompt: str,
        n_sample_frame: int = 8,
        sampling_rate: int = 1,
        stride: int = 1,
        image_mode: str = "RGB",
        image_size = 512,
    ):
        self.path = path
        self.images = self.get_image_list(path)
        self.n_images = len(self.images)

        self.n_sample_frame = n_sample_frame
        self.sampling_rate = sampling_rate

        self.sequence_length = (n_sample_frame - 1) * sampling_rate + 1
        if self.n_images < self.sequence_length:
            raise ValueError

        self.stride = stride
        self.image_mode = image_mode
        self.image_size = image_size

        self.tokenizer = tokenizer
        self.prompt = prompt
        self.prompt_ids = self.tokenizer(
            self.prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

    def __len__(self):
        return (self.n_images - self.sequence_length) // self.stride + 1

    def __getitem__(self, index):
        frame_indices = self.get_frame_indices(index)
        frames = [self.load_frame(i) for i in frame_indices]
        frames = self.transform(frames)

        return {
            "images": frames,
            "prompt_ids": self.prompt_ids,
        }

    def transform(self, frames):
        frames = self.tensorize_frames(frames)
        frames = short_size_scale(frames, size=self.image_size)
        frames = random_crop(frames, height=self.image_size, width=self.image_size)
        return frames

    @staticmethod
    def tensorize_frames(frames):
        frames = rearrange(np.stack(frames), "f h w c -> c f h w")
        return torch.from_numpy(frames).div(255) * 2 - 1


    def load_frame(self, index):
        image_path = os.path.join(self.path, self.images[index])
        return Image.open(image_path).convert(self.image_mode)

    def get_frame_indices(self, index):
        frame_start = self.stride * index
        return (frame_start + i * self.sampling_rate for i in range(self.n_sample_frame))

    @staticmethod
    def get_image_list(path):
        images = []
        for file in sorted(os.listdir(path)):
            if file.endswith(IMAGE_EXTENSION):
                images.append(file)
        return images
