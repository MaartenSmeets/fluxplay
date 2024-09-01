import requests
import json
import time
import uuid
import websocket
import urllib.request
import urllib.parse
import random
import logging
import traceback
import hashlib
import subprocess

# Configuration parameters
API_TYPE = "sbwebui"  # Options: "comfyui" or "sdwebui"
COMFYUI_SERVER_WS_URL = "ws://127.0.0.1:8188"
COMFYUI_SERVER_API_URL = "http://127.0.0.1:8188"
SDWEBUI_SERVER_URL = "http://127.0.0.1:7860"
SDWEBUI_LORA = ""
OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "???"
SDWEBUI_MODEL = "???"
NUM_IMAGES = 30
BATCH_SIZE = 5
TEMPERATURE = 0.7
CLIENT_ID = str(uuid.uuid4())
USE_PROMPT_DIVERSIFICATION = True
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
UNET_MODEL = "flux1-dev.safetensors"
SAMPLER_NAME = "DPM++ 2M SDE Karras"
ADETAILER_DENOISE_STRENGTH = 0.6

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Seed the random number generator with the current time
random.seed(time.time())

# ComfyUI default workflow prompt
comfyui_prompt_text = """
{
  "6": {
    "inputs": {
      "text": "",
      "clip": [
        "11",
        0
      ]
    },
    "class_type": "CLIPTextEncode",
    "_meta": {
      "title": "CLIP Text Encode (Positive Prompt)"
    }
  },
  "8": {
    "inputs": {
      "samples": [
        "13",
        0
      ],
      "vae": [
        "10",
        0
      ]
    },
    "class_type": "VAEDecode",
    "_meta": {
      "title": "VAE Decode"
    }
  },
  "9": {
    "inputs": {
      "filename_prefix": "ComfyUI",
      "images": [
        "8",
        0
      ]
    },
    "class_type": "SaveImage",
    "_meta": {
      "title": "Save Image"
    }
  },
  "10": {
    "inputs": {
      "vae_name": "ae.sft"
    },
    "class_type": "VAELoader",
    "_meta": {
      "title": "Load VAE"
    }
  },
  "11": {
    "inputs": {
      "clip_name1": "t5xxl_fp16.safetensors",
      "clip_name2": "clip_l.safetensors",
      "type": "flux"
    },
    "class_type": "DualCLIPLoader",
    "_meta": {
      "title": "DualCLIPLoader"
    }
  },
  "12": {
    "inputs": {
      "unet_name": "flux1-dev.safetensors",
      "weight_dtype": "default"
    },
    "class_type": "UNETLoader",
    "_meta": {
      "title": "Load Diffusion Model"
    }
  },
  "13": {
    "inputs": {
      "noise": [
        "25",
        0
      ],
      "guider": [
        "22",
        0
      ],
      "sampler": [
        "16",
        0
      ],
      "sigmas": [
        "17",
        0
      ],
      "latent_image": [
        "27",
        0
      ]
    },
    "class_type": "SamplerCustomAdvanced",
    "_meta": {
      "title": "SamplerCustomAdvanced"
    }
  },
  "16": {
    "inputs": {
      "sampler_name": "euler"
    },
    "class_type": "KSamplerSelect",
    "_meta": {
      "title": "KSamplerSelect"
    }
  },
  "17": {
    "inputs": {
      "scheduler": "simple",
      "steps": 28,
      "denoise": 1,
      "model": [
        "30",
        0
      ]
    },
    "class_type": "BasicScheduler",
    "_meta": {
      "title": "BasicScheduler"
    }
  },
  "22": {
    "inputs": {
      "model": [
        "30",
        0
      ],
      "conditioning": [
        "26",
        0
      ]
    },
    "class_type": "BasicGuider",
    "_meta": {
      "title": "BasicGuider"
    }
  },
  "25": {
    "inputs": {
      "noise_seed": 91432082492604
    },
    "class_type": "RandomNoise",
    "_meta": {
      "title": "RandomNoise"
    }
  },
  "26": {
    "inputs": {
      "guidance": 3.5,
      "conditioning": [
        "6",
        0
      ]
    },
    "class_type": "FluxGuidance",
    "_meta": {
      "title": "FluxGuidance"
    }
  },
  "27": {
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    },
    "class_type": "EmptySD3LatentImage",
    "_meta": {
      "title": "EmptySD3LatentImage"
    }
  },
  "30": {
    "inputs": {
      "max_shift": 1.15,
      "base_shift": 0.5,
      "width": 1024,
      "height": 1024,
      "model": [
        "12",
        0
      ]
    },
    "class_type": "ModelSamplingFlux",
    "_meta": {
      "title": "ModelSamplingFlux"
    }
  }
}
"""

sdwebui_payload_templates = {
    "flux": """{
        "prompt": "",
        "steps": 28,
        "sampler_name": "Euler a",
        "cfg_scale": 1.0,
        "width": 1024,
        "height": 1024,
        "negative_prompt": "",
        "seed": -1,
        "override_settings": {
            "sd_model_checkpoint": ""
        },
        "override_settings_restore_afterwards": true,
        "save_images": true,
        "alwayson_scripts": {
            "ADetailer": {
                "args": [
                    true,
                    false,
                    {
                        "ad_cfg_scale": 7,
                        "ad_checkpoint": "Use same checkpoint",
                        "ad_clip_skip": 1,
                        "ad_confidence": 0.3,
                        "ad_controlnet_guidance_end": 1,
                        "ad_controlnet_guidance_start": 0,
                        "ad_controlnet_model": "None",
                        "ad_controlnet_module": "None",
                        "ad_controlnet_weight": 1,
                        "ad_denoising_strength": 0.4,
                        "ad_dilate_erode": 4,
                        "ad_inpaint_height": 512,
                        "ad_inpaint_only_masked": true,
                        "ad_inpaint_only_masked_padding": 32,
                        "ad_inpaint_width": 512,
                        "ad_mask_blur": 4,
                        "ad_mask_k_largest": 0,
                        "ad_mask_max_ratio": 1,
                        "ad_mask_merge_invert": "None",
                        "ad_mask_min_ratio": 0,
                        "ad_model": "face_yolov8n.pt",
                        "ad_model_classes": "",
                        "ad_negative_prompt": "",
                        "ad_noise_multiplier": 1,
                        "ad_prompt": "",
                        "ad_restore_face": false,
                        "ad_sampler": "DPM++ 2M",
                        "ad_scheduler": "Use same scheduler",
                        "ad_steps": 28,
                        "ad_tab_enable": true,
                        "ad_use_cfg_scale": false,
                        "ad_use_checkpoint": false,
                        "ad_use_clip_skip": false,
                        "ad_use_inpaint_width_height": false,
                        "ad_use_noise_multiplier": false,
                        "ad_use_sampler": false,
                        "ad_use_steps": false,
                        "ad_use_vae": false,
                        "ad_vae": "Use same VAE",
                        "ad_x_offset": 0,
                        "ad_y_offset": 0,
                        "is_api": []
                    },
                    {
                        "ad_cfg_scale": 7,
                        "ad_checkpoint": "Use same checkpoint",
                        "ad_clip_skip": 1,
                        "ad_confidence": 0.3,
                        "ad_controlnet_guidance_end": 1,
                        "ad_controlnet_guidance_start": 0,
                        "ad_controlnet_model": "None",
                        "ad_controlnet_module": "None",
                        "ad_controlnet_weight": 1,
                        "ad_denoising_strength": 0.4,
                        "ad_dilate_erode": 4,
                        "ad_inpaint_height": 512,
                        "ad_inpaint_only_masked": true,
                        "ad_inpaint_only_masked_padding": 32,
                        "ad_inpaint_width": 512,
                        "ad_mask_blur": 4,
                        "ad_mask_k_largest": 0,
                        "ad_mask_max_ratio": 1,
                        "ad_mask_merge_invert": "None",
                        "ad_mask_min_ratio": 0,
                        "ad_model": "None",
                        "ad_model_classes": "",
                        "ad_negative_prompt": "",
                        "ad_noise_multiplier": 1,
                        "ad_prompt": "",
                        "ad_restore_face": false,
                        "ad_sampler": "DPM++ 2M",
                        "ad_scheduler": "Use same scheduler",
                        "ad_steps": 28,
                        "ad_tab_enable": false,
                        "ad_use_cfg_scale": false,
                        "ad_use_checkpoint": false,
                        "ad_use_clip_skip": false,
                        "ad_use_inpaint_width_height": false,
                        "ad_use_noise_multiplier": false,
                        "ad_use_sampler": false,
                        "ad_use_steps": false,
                        "ad_use_vae": false,
                        "ad_vae": "Use same VAE",
                        "ad_x_offset": 0,
                        "ad_y_offset": 0,
                        "is_api": []
                    },
                    {
                        "ad_cfg_scale": 7,
                        "ad_checkpoint": "Use same checkpoint",
                        "ad_clip_skip": 1,
                        "ad_confidence": 0.3,
                        "ad_controlnet_guidance_end": 1,
                        "ad_controlnet_guidance_start": 0,
                        "ad_controlnet_model": "None",
                        "ad_controlnet_module": "None",
                        "ad_controlnet_weight": 1,
                        "ad_denoising_strength": 0.4,
                        "ad_dilate_erode": 4,
                        "ad_inpaint_height": 512,
                        "ad_inpaint_only_masked": true,
                        "ad_inpaint_only_masked_padding": 32,
                        "ad_inpaint_width": 512,
                        "ad_mask_blur": 4,
                        "ad_mask_k_largest": 0,
                        "ad_mask_max_ratio": 1,
                        "ad_mask_merge_invert": "None",
                        "ad_mask_min_ratio": 0,
                        "ad_model": "None",
                        "ad_model_classes": "",
                        "ad_negative_prompt": "",
                        "ad_noise_multiplier": 1,
                        "ad_prompt": "",
                        "ad_restore_face": false,
                        "ad_sampler": "DPM++ 2M",
                        "ad_scheduler": "Use same scheduler",
                        "ad_steps": 28,
                        "ad_tab_enable": false,
                        "ad_use_cfg_scale": false,
                        "ad_use_checkpoint": false,
                        "ad_use_clip_skip": false,
                        "ad_use_inpaint_width_height": false,
                        "ad_use_noise_multiplier": false,
                        "ad_use_sampler": false,
                        "ad_use_steps": false,
                        "ad_use_vae": false,
                        "ad_vae": "Use same VAE",
                        "ad_x_offset": 0,
                        "ad_y_offset": 0,
                        "is_api": []
                    },
                    {
                        "ad_cfg_scale": 7,
                        "ad_checkpoint": "Use same checkpoint",
                        "ad_clip_skip": 1,
                        "ad_confidence": 0.3,
                        "ad_controlnet_guidance_end": 1,
                        "ad_controlnet_guidance_start": 0,
                        "ad_controlnet_model": "None",
                        "ad_controlnet_module": "None",
                        "ad_controlnet_weight": 1,
                        "ad_denoising_strength": 0.4,
                        "ad_dilate_erode": 4,
                        "ad_inpaint_height": 512,
                        "ad_inpaint_only_masked": true,
                        "ad_inpaint_only_masked_padding": 32,
                        "ad_inpaint_width": 512,
                        "ad_mask_blur": 4,
                        "ad_mask_k_largest": 0,
                        "ad_mask_max_ratio": 1,
                        "ad_mask_merge_invert": "None",
                        "ad_mask_min_ratio": 0,
                        "ad_model": "None",
                        "ad_model_classes": "",
                        "ad_negative_prompt": "",
                        "ad_noise_multiplier": 1,
                        "ad_prompt": "",
                        "ad_restore_face": false,
                        "ad_sampler": "DPM++ 2M",
                        "ad_scheduler": "Use same scheduler",
                        "ad_steps": 28,
                        "ad_tab_enable": false,
                        "ad_use_cfg_scale": false,
                        "ad_use_checkpoint": false,
                        "ad_use_clip_skip": false,
                        "ad_use_inpaint_width_height": false,
                        "ad_use_noise_multiplier": false,
                        "ad_use_sampler": false,
                        "ad_use_steps": false,
                        "ad_use_vae": false,
                        "ad_vae": "Use same VAE",
                        "ad_x_offset": 0,
                        "ad_y_offset": 0,
                        "is_api": []
                    }
                ]
            }
        }
    }""",
    "sdxl": """{
        "prompt": "",
        "steps": 30,
        "sampler_name": "DPM++ 3M SDE Karras",
        "cfg_scale": 4.0,
        "width": 1024,
        "height": 1024,
        "negative_prompt": "",
        "seed": -1,
        "override_settings": {
            "sd_model_checkpoint": ""
        },
        "override_settings_restore_afterwards": true,
        "save_images": true,
        "alwayson_scripts": {
            "ADetailer": {
                "args": [
                    true,
                    false,
                    {
                        "ad_cfg_scale": 7,
                        "ad_checkpoint": "Use same checkpoint",
                        "ad_clip_skip": 1,
                        "ad_confidence": 0.3,
                        "ad_controlnet_guidance_end": 1,
                        "ad_controlnet_guidance_start": 0,
                        "ad_controlnet_model": "None",
                        "ad_controlnet_module": "None",
                        "ad_controlnet_weight": 1,
                        "ad_denoising_strength": 0.4,
                        "ad_dilate_erode": 4,
                        "ad_inpaint_height": 512,
                        "ad_inpaint_only_masked": true,
                        "ad_inpaint_only_masked_padding": 32,
                        "ad_inpaint_width": 512,
                        "ad_mask_blur": 4,
                        "ad_mask_k_largest": 0,
                        "ad_mask_max_ratio": 1,
                        "ad_mask_merge_invert": "None",
                        "ad_mask_min_ratio": 0,
                        "ad_model": "face_yolov8n.pt",
                        "ad_model_classes": "",
                        "ad_negative_prompt": "",
                        "ad_noise_multiplier": 1,
                        "ad_prompt": "",
                        "ad_restore_face": false,
                        "ad_sampler": "DPM++ 2M",
                        "ad_scheduler": "Use same scheduler",
                        "ad_steps": 28,
                        "ad_tab_enable": true,
                        "ad_use_cfg_scale": false,
                        "ad_use_checkpoint": false,
                        "ad_use_clip_skip": false,
                        "ad_use_inpaint_width_height": false,
                        "ad_use_noise_multiplier": false,
                        "ad_use_sampler": false,
                        "ad_use_steps": false,
                        "ad_use_vae": false,
                        "ad_vae": "Use same VAE",
                        "ad_x_offset": 0,
                        "ad_y_offset": 0,
                        "is_api": []
                    },
                    {
                        "ad_cfg_scale": 7,
                        "ad_checkpoint": "Use same checkpoint",
                        "ad_clip_skip": 1,
                        "ad_confidence": 0.3,
                        "ad_controlnet_guidance_end": 1,
                        "ad_controlnet_guidance_start": 0,
                        "ad_controlnet_model": "None",
                        "ad_controlnet_module": "None",
                        "ad_controlnet_weight": 1,
                        "ad_denoising_strength": 0.4,
                        "ad_dilate_erode": 4,
                        "ad_inpaint_height": 512,
                        "ad_inpaint_only_masked": true,
                        "ad_inpaint_only_masked_padding": 32,
                        "ad_inpaint_width": 512,
                        "ad_mask_blur": 4,
                        "ad_mask_k_largest": 0,
                        "ad_mask_max_ratio": 1,
                        "ad_mask_merge_invert": "None",
                        "ad_mask_min_ratio": 0,
                        "ad_model": "None",
                        "ad_model_classes": "",
                        "ad_negative_prompt": "",
                        "ad_noise_multiplier": 1,
                        "ad_prompt": "",
                        "ad_restore_face": false,
                        "ad_sampler": "DPM++ 2M",
                        "ad_scheduler": "Use same scheduler",
                        "ad_steps": 28,
                        "ad_tab_enable": false,
                        "ad_use_cfg_scale": false,
                        "ad_use_checkpoint": false,
                        "ad_use_clip_skip": false,
                        "ad_use_inpaint_width_height": false,
                        "ad_use_noise_multiplier": false,
                        "ad_use_sampler": false,
                        "ad_use_steps": false,
                        "ad_use_vae": false,
                        "ad_vae": "Use same VAE",
                        "ad_x_offset": 0,
                        "ad_y_offset": 0,
                        "is_api": []
                    },
                    {
                        "ad_cfg_scale": 7,
                        "ad_checkpoint": "Use same checkpoint",
                        "ad_clip_skip": 1,
                        "ad_confidence": 0.3,
                        "ad_controlnet_guidance_end": 1,
                        "ad_controlnet_guidance_start": 0,
                        "ad_controlnet_model": "None",
                        "ad_controlnet_module": "None",
                        "ad_controlnet_weight": 1,
                        "ad_denoising_strength": 0.4,
                        "ad_dilate_erode": 4,
                        "ad_inpaint_height": 512,
                        "ad_inpaint_only_masked": true,
                        "ad_inpaint_only_masked_padding": 32,
                        "ad_inpaint_width": 512,
                        "ad_mask_blur": 4,
                        "ad_mask_k_largest": 0,
                        "ad_mask_max_ratio": 1,
                        "ad_mask_merge_invert": "None",
                        "ad_mask_min_ratio": 0,
                        "ad_model": "None",
                        "ad_model_classes": "",
                        "ad_negative_prompt": "",
                        "ad_noise_multiplier": 1,
                        "ad_prompt": "",
                        "ad_restore_face": false,
                        "ad_sampler": "DPM++ 2M",
                        "ad_scheduler": "Use same scheduler",
                        "ad_steps": 28,
                        "ad_tab_enable": false,
                        "ad_use_cfg_scale": false,
                        "ad_use_checkpoint": false,
                        "ad_use_clip_skip": false,
                        "ad_use_inpaint_width_height": false,
                        "ad_use_noise_multiplier": false,
                        "ad_use_sampler": false,
                        "ad_use_steps": false,
                        "ad_use_vae": false,
                        "ad_vae": "Use same VAE",
                        "ad_x_offset": 0,
                        "ad_y_offset": 0,
                        "is_api": []
                    },
                    {
                        "ad_cfg_scale": 7,
                        "ad_checkpoint": "Use same checkpoint",
                        "ad_clip_skip": 1,
                        "ad_confidence": 0.3,
                        "ad_controlnet_guidance_end": 1,
                        "ad_controlnet_guidance_start": 0,
                        "ad_controlnet_model": "None",
                        "ad_controlnet_module": "None",
                        "ad_controlnet_weight": 1,
                        "ad_denoising_strength": 0.4,
                        "ad_dilate_erode": 4,
                        "ad_inpaint_height": 512,
                        "ad_inpaint_only_masked": true,
                        "ad_inpaint_only_masked_padding": 32,
                        "ad_inpaint_width": 512,
                        "ad_mask_blur": 4,
                        "ad_mask_k_largest": 0,
                        "ad_mask_max_ratio": 1,
                        "ad_mask_merge_invert": "None",
                        "ad_mask_min_ratio": 0,
                        "ad_model": "None",
                        "ad_model_classes": "",
                        "ad_negative_prompt": "",
                        "ad_noise_multiplier": 1,
                        "ad_prompt": "",
                        "ad_restore_face": false,
                        "ad_sampler": "DPM++ 2M",
                        "ad_scheduler": "Use same scheduler",
                        "ad_steps": 28,
                        "ad_tab_enable": false,
                        "ad_use_cfg_scale": false,
                        "ad_use_checkpoint": false,
                        "ad_use_clip_skip": false,
                        "ad_use_inpaint_width_height": false,
                        "ad_use_noise_multiplier": false,
                        "ad_use_sampler": false,
                        "ad_use_steps": false,
                        "ad_use_vae": false,
                        "ad_vae": "Use same VAE",
                        "ad_x_offset": 0,
                        "ad_y_offset": 0,
                        "is_api": []
                    }
                ]
            }
        }
    }"""
}


def determine_template_type(model_name):
    """Determine the template type based on the model name."""
    model_name_lower = model_name.lower()
    if "flux" in model_name_lower:
        return "flux"
    elif "sd" in model_name_lower or "sdxl" in model_name_lower:
        return "sdxl"
    else:
        return "sdxl"  # Default to sdxl if none match


# Determine the template based on the model name
SDWEBUI_TEMPLATE_TYPE = determine_template_type(SDWEBUI_MODEL)


def queue_prompt_comfyui(prompt):
    """Queue a prompt in ComfyUI."""
    p = {"prompt": prompt, "client_id": CLIENT_ID}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(
        f"{COMFYUI_SERVER_API_URL}/prompt", data=data, headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req).read())


def get_images_comfyui(ws, prompt):
    """Retrieve generated images from ComfyUI."""
    prompt_id = queue_prompt_comfyui(prompt)['prompt_id']
    output_images = {}

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing' and message['data']['prompt_id'] == prompt_id:
                break  # Execution is done

    history = get_history_comfyui(prompt_id)[prompt_id]
    for node_id, node_output in history['outputs'].items():
        if 'images' in node_output:
            images_output = [get_image_comfyui(
                img['filename'], img['subfolder'], img['type']) for img in node_output['images']]
            output_images[node_id] = images_output

    return output_images


def get_image_comfyui(filename, subfolder, folder_type):
    """Retrieve a single image from ComfyUI."""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{COMFYUI_SERVER_API_URL}/view?{url_values}") as response:
        return response.read()


def get_history_comfyui(prompt_id):
    """Retrieve history of a specific prompt in ComfyUI."""
    with urllib.request.urlopen(f"{COMFYUI_SERVER_API_URL}/history/{prompt_id}") as response:
        return json.loads(response.read())


def generate_images_sdwebui(prompt, negative_prompt, batch_size=1):
    """Generate images using SDWebUI."""
    url = f"{SDWEBUI_SERVER_URL}/sdapi/v1/txt2img"
    template = sdwebui_payload_templates[SDWEBUI_TEMPLATE_TYPE]
    payload = json.loads(template)

    # Ensure that prompts are not None and are properly formatted
    safe_prompt = prompt if prompt is not None else ""
    safe_negative_prompt = negative_prompt if negative_prompt is not None else ""

    # Set the main prompt and negative prompt
    payload["prompt"] = f"{safe_prompt}, {SDWEBUI_LORA}".strip()
    payload["negative_prompt"] = safe_negative_prompt

    # Update the ad_prompt and ad_negative_prompt in ADetailer args directly in the payload
    if "alwayson_scripts" in payload and "ADetailer" in payload["alwayson_scripts"]:
        for arg in payload["alwayson_scripts"]["ADetailer"]["args"]:
            if isinstance(arg, dict):
                arg["ad_prompt"] = safe_prompt
                arg["ad_negative_prompt"] = safe_negative_prompt
                arg["ad_inpaint_width"] = IMAGE_WIDTH
                arg["ad_inpaint_height"] = IMAGE_HEIGHT
                arg["ad_sampler"] = SAMPLER_NAME
                arg["ad_denoising_strength"] = ADETAILER_DENOISE_STRENGTH

    payload["batch_size"] = 1  # Always 1 for SDWebUI
    payload["n_iter"] = batch_size
    payload["width"] = IMAGE_WIDTH
    payload["height"] = IMAGE_HEIGHT
    payload["override_settings"]["sd_model_checkpoint"] = SDWEBUI_MODEL
    payload["sampler_name"] = SAMPLER_NAME
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        images = result['images']
        logging.info(f"Generated {batch_size} images for prompt: {safe_prompt}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error during image generation: {e}")
        return []

    return images


def diversify_prompt(base_prompt, llm_model=LLM_MODEL, temperature=TEMPERATURE):
    """Diversify the prompt using an LLM model."""
    original_prompt = (
        f"Utilize your expertise in engineering stable diffusion prompts to create a concise detailed prompt which should be based on the provided base prompt (rephrasing, diversification reordering and elaboration is allowed). Put the most important elements at the start of the prompt. "
        f"Avoid repetition. Focus on clear visual elements and physical objects, and emotional impact on the viewer. Prefer explicit to the point short descriptions."
        f"Present the enhanced prompt in a concise format. Remove redundant and non-essential words from your output such as stop words or filler words. Output only the prompt on a single line. Base prompt: {base_prompt}"
    )

    url = f"{OLLAMA_URL}/api/generate"
    payload = {
        "model": llm_model,
        "prompt": original_prompt,
        "temperature": temperature,
        "stream": False
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        response_json = response.json()

        if 'response' in response_json:
            enhanced_prompt = response_json['response'].strip('{} \n')
            subprocess.call("./restart_ollama.sh", shell=True)
            return enhanced_prompt
        else:
            logging.error("No 'response' field in the API response.")
            return base_prompt

    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return base_prompt  # Fallback to base prompt on error


def generate_filename_from_prompt(prompt_text, batch_number):
    """Generate a filename based on the prompt text and batch number."""
    prompt_hash = hashlib.md5(prompt_text.encode('utf-8')).hexdigest()[:8]
    return f"batch_{batch_number}_prompt_{prompt_hash}"


def generate_multiple_images(prompt, negative_prompt, num_images=NUM_IMAGES, batch_size=BATCH_SIZE, llm_model=LLM_MODEL):
    """Generate multiple images using either ComfyUI or SDWebUI."""
    if API_TYPE == "comfyui":
        ws = websocket.create_connection(
            f"{COMFYUI_SERVER_WS_URL}/ws?clientId={CLIENT_ID}")

    original_prompt_text = prompt["6"]["inputs"]["text"] if API_TYPE == "comfyui" else prompt

    for i in range(0, num_images, batch_size):
        current_batch_size = min(batch_size, num_images - i)

        # Diversify the prompt if needed
        if USE_PROMPT_DIVERSIFICATION:
            diversified_prompt = diversify_prompt(
                original_prompt_text, llm_model)
            logging.info(
                f"Using diversified prompt for batch {i // batch_size + 1}: {diversified_prompt}")
        else:
            diversified_prompt = original_prompt_text
            logging.info(
                f"Using original prompt for batch {i // batch_size + 1}: {original_prompt_text}")

        if API_TYPE == "comfyui":
            prompt["6"]["inputs"]["text"] = diversified_prompt
            prompt["9"]["inputs"]["filename_prefix"] = generate_filename_from_prompt(
                original_prompt_text, i // batch_size + 1)
            prompt["25"]["inputs"]["noise_seed"] = random.getrandbits(64)
            prompt["27"]["inputs"]["batch_size"] = current_batch_size
            prompt["27"]["inputs"]["width"] = IMAGE_WIDTH
            prompt["27"]["inputs"]["height"] = IMAGE_HEIGHT
            prompt["16"]["inputs"]["sampler_name"] = SAMPLER_NAME

            images = get_images_comfyui(ws, prompt)
            logging.info(
                f"Generated {current_batch_size} images for seed {prompt['25']['inputs']['noise_seed']} with filename prefix {prompt['9']['inputs']['filename_prefix']}")
        else:
            # For SDWebUI, use batch_size as the number of iterations
            images = generate_images_sdwebui(
                diversified_prompt, negative_prompt, current_batch_size)
            logging.info(
                f"Generated {current_batch_size} images with prompt: {diversified_prompt}")


# Read the prompt text from a file
with open("prompt_text.txt", "r") as file:
    prompt_text_from_file = file.read().strip()

# Read the negative prompt text from a file, if supported by the model
negative_prompt_text = ""
if SDWEBUI_TEMPLATE_TYPE == "sdxl":
    with open("negative_prompt_text.txt", "r") as file:
        negative_prompt_text = file.read().strip()

# Convert the comfyui_prompt_text string to a Python dictionary using json.loads
comfyui_prompt = json.loads(comfyui_prompt_text)

if API_TYPE == "comfyui":
    comfyui_prompt["6"]["inputs"]["text"] = prompt_text_from_file
    comfyui_prompt["27"]["inputs"]["width"] = IMAGE_WIDTH
    comfyui_prompt["27"]["inputs"]["height"] = IMAGE_HEIGHT
    comfyui_prompt["16"]["inputs"]["sampler_name"] = SAMPLER_NAME
    generate_multiple_images(comfyui_prompt, negative_prompt_text)
else:
    generate_multiple_images(prompt_text_from_file, negative_prompt_text)
