# Tune-A-Video

Unofficial implementation of [Tune-A-Video](https://arxiv.org/abs/2212.11565)

![step_500_resized](https://user-images.githubusercontent.com/26464535/211755110-94ba1fbf-a55a-4f71-b665-b7ab2a26121c.gif)

### Training
```
accelerate launch train.py --config config/sample.yml
```

### Notes
* Learning rate and training steps in the [sample config](config/sample.yml) are different from the paper.
* The model seems to memorize a video clip very quickly. `prior_preservation` is added for the regularization.
* Requires > 20GB GPU memory with xformers enabled.

### References
* Pseudo 3D Conv & Temporal Attention: https://github.com/lucidrains/make-a-video-pytorch
* Training Code: https://github.com/huggingface/diffusers
* Video Source: https://youtu.be/RC-24Nfr7fc?t=100


