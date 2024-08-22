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
import re
import hashlib
import subprocess

# Configuration parameters
API_TYPE = "comfyui"  # Options: "comfyui" or "sdwebui"
COMFYUI_SERVER_WS_URL = "ws://127.0.0.1:8188"  # WebSocket requires ws:// or wss://
COMFYUI_SERVER_API_URL = "http://127.0.0.1:8188"  # REST API URL for ComfyUI
SDWEBUI_SERVER_URL = "http://127.0.0.1:7860"
OLLAMA_URL = "http://localhost:11434"  # Configurable Ollama URL
LLM_MODEL = "???"  # Replace with your actual model name
SDWEBUI_MODEL = "???"  # Replace with your desired model name for SDWebUI
UNET_MODEL = "???"
NUM_IMAGES = 30  # Total number of images to generate
BATCH_SIZE = 5  # Number of images processed with the same diversified prompt
TEMPERATURE = 0.7  # Adjust the temperature for creative prompt generation
CLIENT_ID = str(uuid.uuid4())
USE_PROMPT_DIVERSIFICATION = True  # Set to False if prompt diversification is not needed
IMAGE_WIDTH = 1024  # Configurable image width
IMAGE_HEIGHT = 1024  # Configurable image height

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Seed the random number generator with the current time
random.seed(time.time())

# ComfyUI default workflow prompt (as provided in the original code)
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

# Configurable SDWebUI payload templates (as provided in the original code)
sdwebui_payload_templates = {
    "flux": {
        "prompt": "",
        "steps": 28,
        "sampler_name": "Euler a",
        "batch_size": 1,  # Batch size is always 1 for SDWebUI
        "n_iter": 1,
        "cfg_scale": 1.0,
        "distilled_cfg_scale": 3.5,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "negative_prompt": "",
        "seed": -1,
        "override_settings": {
            "sd_model_checkpoint": SDWEBUI_MODEL
        },
        "override_settings_restore_afterwards": True,
        "save_images": True
    },
    "sdxl": {
        "prompt": "",
        "steps": 30,
        "sampler_name": "DPM++ 2M Karras",
        "batch_size": 1,  # Batch size is always 1 for SDWebUI
        "n_iter": 1,
        "cfg_scale": 7.0,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "negative_prompt": "",
        "seed": -1,
        "override_settings": {
            "sd_model_checkpoint": SDWEBUI_MODEL
        },
        "override_settings_restore_afterwards": True,
        "save_images": True
    }
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
    p = {"prompt": prompt, "client_id": CLIENT_ID}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(f"{COMFYUI_SERVER_API_URL}/prompt", data=data, headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(req).read())

def get_image_comfyui(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{COMFYUI_SERVER_API_URL}/view?{url_values}") as response:
        return response.read()

def get_history_comfyui(prompt_id):
    with urllib.request.urlopen(f"{COMFYUI_SERVER_API_URL}/history/{prompt_id}") as response:
        return json.loads(response.read())

def get_images_comfyui(ws, prompt):
    prompt_id = queue_prompt_comfyui(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break  # Execution is done
        else:
            continue  # previews are binary data

    history = get_history_comfyui(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image_comfyui(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
                output_images[node_id] = images_output

    return output_images

def generate_images_sdwebui(prompt, batch_size=1):
    url = f"{SDWEBUI_SERVER_URL}/sdapi/v1/txt2img"
    payload = sdwebui_payload_templates[SDWEBUI_TEMPLATE_TYPE].copy()
    payload["prompt"] = prompt
    payload["batch_size"] = batch_size  # Number of images to generate per API call

    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    images = result['images']
    logging.info(f"Generated {batch_size} images for prompt: {prompt}")

    return images

def unpack_dict_values(d):
    """Recursively unpack dictionary values into a flat list of strings."""
    result = []
    for value in d.values():
        if isinstance(value, dict):
            result.extend(unpack_dict_values(value))
        else:
            result.append(str(value))
    return result

def clean_json_string(json_str):
    """Cleans up a JSON string by removing unexpected characters and extra whitespace."""
    cleaned_str = re.sub(r'\n\s*\n', '', json_str)  # Remove excessive newlines
    cleaned_str = re.sub(r'\s+', ' ', cleaned_str).strip()  # Collapse whitespace
    return cleaned_str

def diversify_prompt(base_prompt, llm_model=LLM_MODEL, temperature=TEMPERATURE):
    original_prompt = (
        f"Utilize your expertise in engineering stable diffusion prompts to create an elaborate, "
        f"highly detailed, and unique prompt which should be based on the provided base prompt (rephrasing, diversification and elaboration is allowed). "
        f"Avoid repetition. Focus on clear visual elements, expressiveness, and emotional impact, with a strong emphasis on precision and creative "
        f"variation in style and colors. The generated prompt should be a thoughtful, complex creation. Prefer explicit descriptions."
        f"Present the enhanced prompt in a concise format. Remove redundant and non essential words from your output such as stop words or filler words. Output only the prompt on a single line. Base prompt: {base_prompt}"
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
        response.raise_for_status()  # Will raise an error for HTTP errors
        response_json = response.json()

        # Logging for debugging
        logging.debug(f"Response status code: {response.status_code}")
        logging.debug(f"Response content: {json.dumps(response_json, indent=2)}")

        # Parsing the response
        if 'response' in response_json:
            enhanced_prompt_json_str = response_json['response']
            try:
                # Parse the JSON response if it's in JSON format
                enhanced_prompt_dict = json.loads(enhanced_prompt_json_str)
                flat_prompt_list = unpack_dict_values(enhanced_prompt_dict)
                final_prompt = ', '.join(flat_prompt_list).strip().replace(' ,', ',').replace(' .', '.')
            except json.JSONDecodeError:
                logging.warning("Response content is not valid JSON. Using as plain text.")
                final_prompt = enhanced_prompt_json_str.strip('{} \n')
            
            ## Call the restart script after the API call
            subprocess.call("./restart_ollama.sh", shell=True)

            return final_prompt
        else:
            logging.error("No 'response' field in the API response.")
            return base_prompt

    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {str(e)}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return base_prompt  # Fallback to base prompt on error

def generate_filename_from_prompt(prompt_text, batch_number):
    """Generate a filename based on the prompt text and batch number."""
    # Create a hash of the prompt text to make the filename unique and concise
    prompt_hash = hashlib.md5(prompt_text.encode('utf-8')).hexdigest()[:8]
    filename_prefix = f"batch_{batch_number}_prompt_{prompt_hash}"
    return filename_prefix

def generate_multiple_images(prompt, num_images=NUM_IMAGES, batch_size=BATCH_SIZE, llm_model=LLM_MODEL):
    if API_TYPE == "comfyui":
        ws = websocket.create_connection(f"{COMFYUI_SERVER_WS_URL}/ws?clientId={CLIENT_ID}")
    
    # Store the original prompt text
    original_prompt_text = prompt["6"]["inputs"]["text"] if API_TYPE == "comfyui" else prompt
    
    for i in range(0, num_images, batch_size):
        current_batch_size = min(batch_size, num_images - i)
        
        # Diversify the prompt once for the current batch
        if USE_PROMPT_DIVERSIFICATION:
            diversified_prompt = diversify_prompt(original_prompt_text, llm_model)
            logging.info(f"Using diversified prompt for batch {i // batch_size + 1}: {diversified_prompt}")
        else:
            diversified_prompt = original_prompt_text
            logging.info(f"Using original prompt for batch {i // batch_size + 1}: {original_prompt_text}")

        if API_TYPE == "comfyui":
            prompt["6"]["inputs"]["text"] = diversified_prompt
            prompt["9"]["inputs"]["filename_prefix"] = generate_filename_from_prompt(original_prompt_text, i // batch_size + 1)
            prompt["25"]["inputs"]["noise_seed"] = random.getrandbits(64)
            prompt["27"]["inputs"]["batch_size"] = current_batch_size

            images = get_images_comfyui(ws, prompt)
            logging.info(f"Generated {current_batch_size} images for seed {prompt['25']['inputs']['noise_seed']} with filename prefix {prompt['9']['inputs']['filename_prefix']}")
        else:
            # For SDWebUI, generate one image at a time in a loop
            images = []
            for _ in range(current_batch_size):
                images.extend(generate_images_sdwebui(diversified_prompt))
            logging.info(f"Generated {current_batch_size} images with prompt: {diversified_prompt}")

# Read the prompt text from a file
with open("prompt_text.txt", "r") as file:
    prompt_text_from_file = file.read().strip()

if API_TYPE == "comfyui":
    prompt = json.loads(comfyui_prompt_text)
    prompt["6"]["inputs"]["text"] = prompt_text_from_file
    prompt["27"]["inputs"]["width"] = IMAGE_WIDTH
    prompt["27"]["inputs"]["height"] = IMAGE_HEIGHT
else:
    prompt = prompt_text_from_file

generate_multiple_images(prompt)
