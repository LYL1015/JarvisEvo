# -*- coding: utf-8 -*-
from PIL import Image
from diffusers import QwenImageEditPipeline
import torch
import random


class AIGCManager:
    def __init__(self, model_ckpt=None, device="cpu"):
        print("load model")
        self.pipeline = QwenImageEditPipeline.from_pretrained(model_ckpt)
        self.pipeline.to(torch.bfloat16)
        self.pipeline.to(device)
        self.pipeline.set_progress_bar_config(disable=None)

    def call_img2img_model(
            self, 
            image_path,
            prompt,
            save_path,
            **kwards
        ):

        inputs = {
            "image": Image.open(image_path).convert("RGB"),
            "prompt": prompt,
            "generator": torch.manual_seed(random.randint(0, 10000)),
            "true_cfg_scale": 4.0,
            "negative_prompt": " ",
            "num_inference_steps": 50,
        }

        with torch.inference_mode():
            output = self.pipeline(**inputs)
            output_image = output.images[0]
            output_image.save(save_path)
            print(f"aigc_path: {save_path}")

        return save_path
    
    def call_img2img(
            self, 
            image_path,
            prompt,
            save_path,
        ):

        return self.call_img2img_model(image_path, prompt, save_path)
