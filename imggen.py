import argparse
from math import gcd
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import numpy as np
from PIL import Image
import torch
import os
from image_to_image import StableDiffusionImg2ImgPipeline
from image_to_image import preprocess as preprocess_init

MY_TOKEN = os.environ["HF_TOKEN"]

def generate_variations(prompt, num_samples, gc, steps, height=512, width=768, seed=None, init_image=None):
    # Load model
    model_id = "CompVis/stable-diffusion-v1-4"
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    pipelinekind = StableDiffusionImg2ImgPipeline if init_image else StableDiffusionPipeline
    pipe = pipelinekind.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32, use_auth_token=MY_TOKEN)
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    pipe.to("cuda")
    # Load init image (if provided)
    if init_image:
        x0 = Image.open(init_image).convert("RGB")
        x0 = x0.resize((height, width))
        x0 = preprocess_init(x0)

    ## Generate variations
    for _ in range(num_samples):
        seed_image = seed if seed else np.random.randint(9999999999)
        pipeargs = {
            "guidance_scale": gc,
            "num_inference_steps": steps,
            "prompt": prompt,
        }
        if init_image:
            pipeargs = {**pipeargs, "init_image": x0}
        else:
            pipeargs = {**pipeargs, "height": height, "width": width}
        image = pipe(**pipeargs)["sample"][0]
        image.save(f"{height}x{width}_seed{seed_image}_gc{gc}_steps{steps}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate stable diffusion variations.')
    parser.add_argument('prompt', type=str, help='Prompt for generation')
    parser.add_argument("-n","--num_samples", type=int, default=1, help="How many samples to produce (default: 1)")
    parser.add_argument('--height', type=int, default=512, help='Height in pixels of generated image')
    parser.add_argument('--width', type=int, default=512, help='Width in pixels of generated image')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for all generations (randomize seed if None)')
    parser.add_argument('--gc', type=float, default=7.5, help='Classifier-free Guidance scale  (default: 7.5)')
    parser.add_argument('--steps', type=int, default=50, help='Number of diffusion steps (default: 50)')
    parser.add_argument('--init_image', type=str, default=None, help='Path to an image to use as initialization')
    args = parser.parse_args()
    generate_variations(
        prompt=args.prompt,
        num_samples=args.num_samples,
        gc=args.gc,
        steps=args.steps,
        height=args.height,
        width=args.width,
        seed=args.seed,
        init_image=args.init_image,
    )
