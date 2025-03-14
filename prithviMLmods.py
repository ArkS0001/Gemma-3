import os
import random
import uuid
import json
import time
import asyncio
import re
from threading import Thread

# Removed: import gradio as gr
# Removed: import spaces

import torch
import numpy as np
from PIL import Image
import edge_tts
import cv2

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    Gemma3ForConditionalGeneration,
)
from transformers.image_utils import load_image
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

# --------------------------------------------------------------------------------
# CONFIG & CONSTANTS
# --------------------------------------------------------------------------------

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))
MAX_SEED = np.iinfo(np.int32).max

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Removed the progress_bar_html function (UI-specific)

# TEXT & TTS MODELS
model_id = "prithivMLmods/FastThink-0.5B-Tiny"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model.eval()

TTS_VOICES = [
    "en-US-JennyNeural",  # @tts1
    "en-US-GuyNeural",    # @tts2
]

# MULTIMODAL (OCR) MODELS
MODEL_ID_VL = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct"
processor = AutoProcessor.from_pretrained(MODEL_ID_VL, trust_remote_code=True)
model_m = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID_VL,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# TTS HELPER
async def text_to_speech(text: str, voice: str, output_file="output.mp3"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)
    return output_file

def clean_chat_history(chat_history):
    """Ensures each item is a dict with 'content' as a string."""
    cleaned = []
    for msg in chat_history:
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            cleaned.append(msg)
    return cleaned

bad_words = json.loads(os.getenv('BAD_WORDS', "[]"))
bad_words_negative = json.loads(os.getenv('BAD_WORDS_NEGATIVE', "[]"))
default_negative = os.getenv("default_negative", "")

def check_text(prompt, negative=""):
    """Example filter function for restricted words."""
    for i in bad_words:
        if i in prompt:
            return True
    for i in bad_words_negative:
        if i in negative:
            return True
    return False

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    """Utility to randomize seeds."""
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES", "0") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE", "0") == "1"
ENABLE_CPU_OFFLOAD = os.getenv("ENABLE_CPU_OFFLOAD", "0") == "1"

dtype = torch.float16 if device.type == "cuda" else torch.float32

# STABLE DIFFUSION IMAGE GENERATION MODEL
if torch.cuda.is_available():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0_Lightning",
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False
    )
    pipe.text_encoder = pipe.text_encoder.half()
    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    if USE_TORCH_COMPILE:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
else:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0_Lightning",
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False
    ).to(device)

DEFAULT_MODEL = "Lightning 5"
models = {
    "Lightning 5": pipe
}

def save_image(img: Image.Image) -> str:
    """Save an image to disk with a unique name."""
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

# GEMMA3-4B MULTIMODAL MODEL
gemma3_model_id = "google/gemma-3-4b-it"
gemma3_model = Gemma3ForConditionalGeneration.from_pretrained(
    gemma3_model_id, device_map="auto"
).eval()
gemma3_processor = AutoProcessor.from_pretrained(gemma3_model_id)

# VIDEO PROCESSING HELPER
def downsample_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    # Sample 10 evenly spaced frames
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            # Convert from BGR to RGB and then to PIL Image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))
    vidcap.release()
    return frames

# --------------------------------------------------------------------------------
# MAIN FUNCTION WITHOUT UI
# --------------------------------------------------------------------------------

def generate_in_colab(
    text: str,
    chat_history: list[dict] = None,
    files: list[str] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
):
    """
    Processes text/image/video commands, loads appropriate models,
    and returns final output (string, image path, or TTS file path).
    - text: user query (supports prefixes like '@lightningv5', '@gemma3', '@video-infer', '@tts1', '@tts2')
    - chat_history: list of messages if you want a multi-turn conversation
    - files: optional list of local file paths (images or video)
    - returns: Python objects (strings, file paths, etc.) depending on the command
    """
    if chat_history is None:
        chat_history = []
    if files is None:
        files = []

    lower_text = text.lower().strip()
    # -------------------------------------------------------------------------
    # 1) IMAGE GENERATION BRANCH: @lightningv5
    # -------------------------------------------------------------------------
    if lower_text.startswith("@lightningv5"):
        prompt_clean = re.sub(r"@lightningv5", "", text, flags=re.IGNORECASE).strip().strip('"')
        
        width = 1024
        height = 1024
        guidance_scale = 6.0
        seed_val = 0
        randomize_seed_flag = True
        seed_val = int(randomize_seed_fn(seed_val, randomize_seed_flag))
        generator = torch.Generator(device=device).manual_seed(seed_val)

        options = {
            "prompt": prompt_clean,
            "negative_prompt": default_negative,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": 30,
            "generator": generator,
            "num_images_per_prompt": 1,
            "use_resolution_binning": True,
            "output_type": "pil",
        }
        if device.type == "cuda":
            torch.cuda.empty_cache()

        images = models["Lightning 5"](**options).images
        image_path = save_image(images[0])
        # Return the path to the generated image
        return {
            "type": "image_generation",
            "prompt": prompt_clean,
            "image_path": image_path
        }

    # -------------------------------------------------------------------------
    # 2) GEMMA3-4B TEXT & MULTIMODAL BRANCH: @gemma3
    # -------------------------------------------------------------------------
    if lower_text.startswith("@gemma3") and not lower_text.startswith("@video-infer"):
        prompt_clean = re.sub(r"@gemma3", "", text, flags=re.IGNORECASE).strip().strip('"')
        if files:
            # If image files are provided, load them
            images = [load_image(f) for f in files]
            messages = [{
                "role": "user",
                "content": [
                    *[{"type": "image", "image": image} for image in images],
                    {"type": "text", "text": prompt_clean},
                ]
            }]
        else:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt_clean}]}
            ]

        inputs = gemma3_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(gemma3_model.device, dtype=torch.bfloat16)

        streamer = TextIteratorStreamer(
            gemma3_processor.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
        # We collect output in a buffer instead of streaming:
        thread = Thread(target=gemma3_model.generate, kwargs=generation_kwargs)
        thread.start()
        buffer = ""
        for new_text in streamer:
            buffer += new_text
        return {
            "type": "gemma3_multimodal",
            "prompt": prompt_clean,
            "response": buffer
        }

    # -------------------------------------------------------------------------
    # 3) GEMMA3-4B VIDEO BRANCH: @video-infer
    # -------------------------------------------------------------------------
    if lower_text.startswith("@video-infer"):
        prompt_clean = re.sub(r"@video-infer", "", text, flags=re.IGNORECASE).strip().strip('"')
        if files:
            # Assume the first file is a video
            video_path = files[0]
            frames = downsample_video(video_path)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt_clean}]}
            ]
            # Append each frame as an image with a timestamp label
            for frame in frames:
                image, timestamp = frame
                messages[1]["content"].append({"type": "text", "text": f"Frame {timestamp}:"})
                # We don't strictly need to save each frame to disk in Colab,
                # but you could if you want. We'll just pass the PIL image object:
                messages[1]["content"].append({"type": "image", "image": image})
        else:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt_clean}]}
            ]

        inputs = gemma3_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(gemma3_model.device, dtype=torch.bfloat16)

        streamer = TextIteratorStreamer(
            gemma3_processor.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        }
        thread = Thread(target=gemma3_model.generate, kwargs=generation_kwargs)
        thread.start()
        buffer = ""
        for new_text in streamer:
            buffer += new_text
        return {
            "type": "gemma3_video",
            "prompt": prompt_clean,
            "response": buffer
        }

    # -------------------------------------------------------------------------
    # 4) TEXT/TTS BRANCH: @tts1 / @tts2 or normal text
    # -------------------------------------------------------------------------
    tts_prefix = "@tts"
    is_tts = any(text.strip().lower().startswith(f"{tts_prefix}{i}") for i in range(1, 3))
    voice_index = next((i for i in range(1, 3) if text.strip().lower().startswith(f"{tts_prefix}{i}")), None)

    if is_tts and voice_index:
        voice = TTS_VOICES[voice_index - 1]
        text_clean = text.replace(f"{tts_prefix}{voice_index}", "").strip()
        conversation = [{"role": "user", "content": text_clean}]
    else:
        voice = None
        text_clean = text.replace(tts_prefix, "").strip()
        conversation = clean_chat_history(chat_history)
        conversation.append({"role": "user", "content": text_clean})

    # If images are attached (files), handle with Qwen2-VL OCR:
    if files:
        images = [load_image(image) for image in files]
        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": text_clean},
            ]
        }]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True).to("cuda")
        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens
        }
        thread = Thread(target=model_m.generate, kwargs=generation_kwargs)
        thread.start()
        buffer = ""
        for new_text in streamer:
            buffer += new_text
            buffer = buffer.replace("<|im_end|>", "")
        final_response = buffer
    else:
        # Normal text conversation with the small text model
        input_ids = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")
        # Truncate if too long
        if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
            input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        input_ids = input_ids.to(model.device)
        streamer = TextIteratorStreamer(tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
            "num_beams": 1,
            "repetition_penalty": repetition_penalty,
        }
        t = Thread(target=model.generate, kwargs=generation_kwargs)
        t.start()
        outputs = []
        for new_text in streamer:
            outputs.append(new_text)
        final_response = "".join(outputs)

    # If TTS is requested, convert final response to speech
    if is_tts and voice:
        audio_file = asyncio.run(text_to_speech(final_response, voice))
        return {
            "type": "tts",
            "text_input": text_clean,
            "response": final_response,
            "audio_file": audio_file
        }
    else:
        return {
            "type": "text_generation",
            "text_input": text_clean,
            "response": final_response
        }


# --------------------------------------------------------------------------------
# EXAMPLE USAGE (In Colab)
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Example usage in a Colab cell:

    result = generate_in_colab("@lightningv5 A fantasy landscape with floating islands")
    print(result)
    # -> { "type": "image_generation", "prompt": "...", "image_path": "xxxx.png" }

    result = generate_in_colab("@tts1 Who is Nikola Tesla, and why did he die?")
    print(result)
    # -> { "type": "tts", "text_input": "...", "response": "...", "audio_file": "output.mp3" }
    """
    # Simple text example
    out = generate_in_colab("Hello, how are you?")
    print("Output:", out)
