from typing import Dict, List, Optional, Union
import multiprocessing
# Set spawn method for CUDA multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from vllm import LLM, SamplingParams
import torch
import vllm
import math
import os
from vllm.assets.image import ImageAsset
from transformers import AutoTokenizer
from PIL import Image
from vllm.multimodal import profiling
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import numpy as np
import io
from PIL import Image
from finetune_src.training.data import get_image_info

def add_gaussian_noise(img, intensity=10):
    """
    Add Gaussian noise to an image with specified intensity.
    
    Args:
        img: PIL Image object
        intensity: Standard deviation of the Gaussian noise (higher = more noise)
                  Range typically 0-50, where 0 means no noise and 50 is very noisy
    
    Returns:
        PIL Image with added noise
    """
    # Convert image to numpy array
    img_array = np.array(img).astype(np.float32)
    
    # Generate Gaussian noise with same shape as image
    noise = np.random.normal(0, intensity, img_array.shape)
    
    # Add noise to image
    noisy_img_array = img_array + noise
    
    # Clip values to valid range [0, 255]
    noisy_img_array = np.clip(noisy_img_array, 0, 255).astype(np.uint8)
    
    # Convert back to PIL Image
    noisy_img = Image.fromarray(noisy_img_array)
    
    return noisy_img

def jpeg_degrade(img, quality):
    with io.BytesIO() as output:
        img.convert('RGB').save(output, format='JPEG', quality=quality)
        output.seek(0)  # Move the reading cursor to the start of the stream
        img_jpeg = Image.open(output).copy()  # Use .copy() to make sure the image is loaded in memory
    return img_jpeg


class VLLMInferenceWrapper:
    """Wrapper class for vLLM inference on vision language models."""
    
    def __init__(
        self,
        model_path: str,
        model_type: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        gaussian_noise_level: int = 0,
        jpeg_degrade_quality: int = 0
    ):
        self.model_type = model_type
        self.model_path = model_path
        
        # Get number of available GPUs
        self.num_gpus = 1
        
        # Initialize model based on type
        if model_type == "qwen-vl":
            self.llm = LLM(
                model=model_path,
                trust_remote_code=True,
                max_model_len=1024,
                max_num_seqs=8,
                # disable_mm_preprocessor_cache=True,
                device="cuda",
                dtype=torch.bfloat16,
                gpu_memory_utilization=0.90,
                tensor_parallel_size=1 # Automatically use all available GPUs
            )
            self.stop_token_ids = None
        elif model_type in ["qwen2-vl", "qwen2.5-vl"]:
            self.llm = LLM(
                model=model_path,
                max_model_len=4096,
                max_num_seqs=16,
                # disable_mm_preprocessor_cache=True,
                device="cuda",
                mm_processor_kwargs={
                    "min_pixels": 256 * 28 * 28,
                    "max_pixels": 1280 * 28 * 28,
                },
                dtype=torch.bfloat16,
                gpu_memory_utilization=0.85,
                tensor_parallel_size=1 # Automatically use all available GPUs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.stop_token_ids = None
        elif model_type == "internvl":
            self.llm = LLM(
                model=model_path,
                trust_remote_code=True,
                max_model_len=4096,
                disable_mm_preprocessor_cache=True,
                device="cuda",
                mm_processor_kwargs={
                    "max_dynamic_patch": 6,
                },
                max_num_batched_tokens=4096,
                dtype=torch.bfloat16,
                gpu_memory_utilization=0.90,
                tensor_parallel_size=1
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
            self.stop_token_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        elif model_type == "llava":
            self.llm = LLM(
                model=model_path,
                trust_remote_code=True,
                max_model_len=4096,
                device="cuda",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            stop_tokens = ["</s>"]
            self.stop_token_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop_token_ids=self.stop_token_ids
        )
        self.gaussian_noise_level = gaussian_noise_level
        self.jpeg_degrade_quality = jpeg_degrade_quality
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def _format_prompt_qwen_vl(self, image_path: str, prompt: str) -> str:
        """Format prompt for Qwen-VL."""
        return f"{prompt}Picture 1: <img></img>\n"

    def _format_prompt_qwen2_vl(self, image_path: str, prompt: str) -> str:
        """Format prompt for Qwen2-VL and Qwen2.5-VL."""
        if self.model_type == "qwen2-vl":
            placeholder = "<|image_pad|>"
        else:  # qwen2.5-vl
            placeholder = "<|image_pad|>"

        formatted_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return formatted_prompt
    
    def _format_prompt_llava(self, image_path: str, prompt: str) -> str:
        """Format prompt for LLaVA."""
        return f"[INST] <image>\n{prompt}[/INST]"

    def _format_prompt_internvl(self, image_path: str, prompt: str) -> str:
        """Format prompt for InternVL."""
        messages = [{
            'role': 'user',
            'content': f"<image>\n{prompt}"
        }]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted_prompt
    
    def _preprocess_image(
        self, image: "ImageObject", image_max_pixels: int, image_min_pixels: int, **kwargs
    ) -> "ImageObject":
        r"""Pre-process a single image."""
        if (image.width * image.height) > image_max_pixels:
            resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < image_min_pixels:
            resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image
    
    def get_multi_modal_input(self, image_path: str):
        """
        return {
            "data": image or video,
            "question": question,
        }
        """
        # Input image and question
        # if self.model_type == "internvl":
        #     image = Image.open(image_path).convert('RGB')
        #     transform = build_transform(input_size=448)
        #     images = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=6)
        #     pixel_values = [transform(image) for image in images]
        #     pixel_values = torch.stack(pixel_values)
        #     return pixel_values
        # else:
        
        image = Image.open(image_path).convert("RGB")
        if self.model_type in ["qwen2-vl", "qwen2.5-vl"]:
            image = get_image_info(image_path, self.min_pixels, self.max_pixels)
        else:
            # image = self._preprocess_image(image, 490*490, 256*256)
            image = image
        # import pdb; pdb.set_trace()
        if self.gaussian_noise_level > 0:
            image = add_gaussian_noise(image, self.gaussian_noise_level)
        if self.jpeg_degrade_quality > 0:
            image = jpeg_degrade(image, self.jpeg_degrade_quality)
        print(image.size)
        return image


    def generate(self, image_path: str, prompt: str) -> str:
        """Generate response for a given image and prompt."""
        # Format prompt based on model type
        if self.model_type == "qwen-vl":
            formatted_prompt = self._format_prompt_qwen_vl(image_path, prompt)
        elif self.model_type == "internvl":
            formatted_prompt = self._format_prompt_internvl(image_path, prompt)
        elif self.model_type == "llava":
            formatted_prompt = self._format_prompt_llava(image_path, prompt)
        else:  # qwen2-vl or qwen2.5-vl
            formatted_prompt = self._format_prompt_qwen2_vl(image_path, prompt)

        # Prepare input for vLLM
        inputs = {
            "prompt": formatted_prompt,
            "multi_modal_data": {
                "image": self.get_multi_modal_input(image_path)
            },
        }
        # import pdb; pdb.set_trace()
        # Generate response
        outputs = self.llm.generate([inputs], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text