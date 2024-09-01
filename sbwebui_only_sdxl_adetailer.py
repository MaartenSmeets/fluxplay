import requests
import json
import time
import uuid
import random
import logging
import traceback
import hashlib
import subprocess

# Configuration parameters
CONFIG_API_TYPE = "sdwebui"  # Only SDWebUI is supported now
CONFIG_SDWEBUI_SERVER_URL = "http://127.0.0.1:7860"
CONFIG_SDWEBUI_LORA = ""
CONFIG_OLLAMA_URL = "http://localhost:11434"
CONFIG_LLM_MODEL = "???"
CONFIG_SDWEBUI_MODEL = "???"
CONFIG_SDWEBUI_ADETAILER_MODEL = "???"
CONFIG_SAMPLER_NAME = "DPM++ 2M SDE Karras"
CONFIG_SAMPLER_ADETAILER_NAME = "DPM++ 2M SDE Karras"
CONFIG_NUM_IMAGES = 30
CONFIG_BATCH_SIZE = 5
CONFIG_TEMPERATURE = 0.7
CONFIG_CLIENT_ID = str(uuid.uuid4())
CONFIG_USE_PROMPT_DIVERSIFICATION = True
CONFIG_IMAGE_WIDTH = 1024
CONFIG_IMAGE_HEIGHT = 1024
CONFIG_ADETAILER_DENOISE_STRENGTH = 0.4

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Seed the random number generator with the current time
random.seed(time.time())

sdwebui_payload_templates = {
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
            "sd_model_checkpoint": "iniverse_v1.safetensors"
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


def switch_model(model_name):
    """Switch the SDWebUI model by first fetching the current options, updating the model, and then posting the updated payload."""
    options_url = f"{CONFIG_SDWEBUI_SERVER_URL}/sdapi/v1/options"

    try:
        # Fetch the current options
        response = requests.get(options_url)
        response.raise_for_status()
        current_options = response.json()

        # Update the model checkpoint in the current options
        current_options["sd_model_checkpoint"] = model_name

        # Post the updated options
        response = requests.post(options_url, json=current_options)
        response.raise_for_status()
        logging.info(f"Successfully switched to model: {model_name}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to switch model to {model_name}: {e}")
        return False
    return True


def save_prompt_to_file(prompt, filename="generated_prompts.txt"):
    """Save the generated prompt to a text file."""
    with open(filename, "a") as file:
        file.write(prompt + "\n")


def generate_images_sdwebui(prompt, negative_prompt, batch_size=1):
    """Generate images using SDWebUI."""
    url = f"{CONFIG_SDWEBUI_SERVER_URL}/sdapi/v1/txt2img"
    
    template = sdwebui_payload_templates["sdxl"]
    payload = json.loads(template)

    # Ensure that prompts are not None and are properly formatted
    safe_prompt = prompt if prompt is not None else ""
    safe_negative_prompt = negative_prompt if negative_prompt is not None else ""

    # Set the main prompt and negative prompt
    payload["prompt"] = f"{safe_prompt}, {CONFIG_SDWEBUI_LORA}".strip()
    payload["negative_prompt"] = safe_negative_prompt

    # Update the ad_prompt and ad_negative_prompt in ADetailer args directly in the payload
    if "alwayson_scripts" in payload and "ADetailer" in payload["alwayson_scripts"]:
        for arg in payload["alwayson_scripts"]["ADetailer"]["args"]:
            if isinstance(arg, dict):
                arg["ad_prompt"] = safe_prompt
                arg["ad_negative_prompt"] = safe_negative_prompt
                arg["ad_inpaint_width"] = CONFIG_IMAGE_WIDTH
                arg["ad_inpaint_height"] = CONFIG_IMAGE_HEIGHT
                arg["ad_sampler"] = CONFIG_SAMPLER_ADETAILER_NAME
                arg["ad_denoising_strength"] = CONFIG_ADETAILER_DENOISE_STRENGTH
                arg["ad_checkpoint"] = CONFIG_SDWEBUI_ADETAILER_MODEL

    payload["batch_size"] = 1  # Always 1 for SDWebUI
    payload["n_iter"] = batch_size
    payload["width"] = CONFIG_IMAGE_WIDTH
    payload["height"] = CONFIG_IMAGE_HEIGHT
    payload["override_settings"]["sd_model_checkpoint"] = CONFIG_SDWEBUI_MODEL
    payload["sampler_name"] = CONFIG_SAMPLER_NAME

    # Attempt the request to the SDWebUI server
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

def diversify_prompt(base_prompt, llm_model=CONFIG_LLM_MODEL, temperature=CONFIG_TEMPERATURE):
    """Diversify the prompt using an LLM model."""
    original_prompt = (
        f"Utilize your expertise in engineering Stable Diffusion XL prompts to create a creative detailed prompt which should be inspired on the provided base prompt without deviating too much from the concept and essence (rephrasing, diversification reordering and elaboration are allowed). Put the most important elements at the start of the prompt. "
        f"Avoid repetition, stop words and filler words. Focus on clear visual elements and physical objects. Try to achieve emotional impact on the viewer. Prefer explicit to the point short descriptions."
        f"Output only the created prompt on a single line. Base prompt: {base_prompt}"
    )

    url = f"{CONFIG_OLLAMA_URL}/api/generate"
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


def generate_multiple_images(prompt, negative_prompt, num_images=CONFIG_NUM_IMAGES, batch_size=CONFIG_BATCH_SIZE, llm_model=CONFIG_LLM_MODEL):
    """Generate multiple images using SDWebUI."""
    original_prompt_text = prompt

    # Ensure the correct model is loaded before generating images
    if not switch_model(CONFIG_SDWEBUI_MODEL):
        logging.error(f"Failed to switch to model {CONFIG_SDWEBUI_MODEL}. Aborting image generation.")
        return

    for i in range(0, num_images, batch_size):
        current_batch_size = min(batch_size, num_images - i)

        # Diversify the prompt if needed
        if CONFIG_USE_PROMPT_DIVERSIFICATION:
            diversified_prompt = diversify_prompt(original_prompt_text, llm_model)
            logging.info(f"Using diversified prompt for batch {i // batch_size + 1}: {diversified_prompt}")
        else:
            diversified_prompt = original_prompt_text
            logging.info(f"Using original prompt for batch {i // batch_size + 1}: {original_prompt_text}")

        # Save the generated prompt to a file
        save_prompt_to_file(diversified_prompt)

        # Generate images using SDWebUI
        images = generate_images_sdwebui(diversified_prompt, negative_prompt, current_batch_size)
        logging.info(f"Generated {current_batch_size} images with prompt: {diversified_prompt}")


# Read the prompt text from a file
with open("prompt_text.txt", "r") as file:
    prompt_text_from_file = file.read().strip()

# Read the negative prompt text from a file, if supported by the model
negative_prompt_text = ""
with open("negative_prompt_text.txt", "r") as file:
    negative_prompt_text = file.read().strip()

# Generate images using the provided prompt and negative prompt
generate_multiple_images(prompt_text_from_file, negative_prompt_text)
