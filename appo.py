import base64
import io
import os
import re
import uuid
import time
import asyncio
import webbrowser
import threading
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, field_validator
from PIL import Image, ImageOps

PROMPTS_DIR = Path("prompts")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

API_KEY = "sk-knHaYFMFjUP06hN4YnCYW1k6RisR04fbxtS0dRMMibO6qwAz"
API_URL = "https://api.gapapi.com/v1/chat/completions"
MODEL = "gemini-3-pro-image-preview"

FINAL_JPEG_QUALITY = 95 
INPUT_JPEG_QUALITY = 75  

HIGH_RES_SIZE = 960  
REF_RES_SIZE = 600    

http_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=40)
    http_client = httpx.AsyncClient(timeout=120.0, follow_redirects=True, limits=limits)
    
    def open_browser():
        time.sleep(1.5)

    threading.Thread(target=open_browser, daemon=True).start()
    print("--- 🚀 TURBO SERVER V2 (STABLE) STARTED 🚀 ---")
    yield
    await http_client.aclose()
    print("--- Server Stopped ---")

app = FastAPI(title="Gemini Pro Turbo V2", lifespan=lifespan)

class ModelSetRequest(BaseModel):
    person_image_url: HttpUrl
    model_image_url: Optional[HttpUrl] = None
    clothing_image_url: Optional[HttpUrl] = None
    bag_image_url: Optional[HttpUrl] = None
    shoes_image_url: Optional[HttpUrl] = None
    accessory_image_url: Optional[HttpUrl] = None
    custom_prompt: Optional[str] = None

    @field_validator('model_image_url', 'clothing_image_url', 'bag_image_url', 'shoes_image_url', 'accessory_image_url', mode='before')
    @classmethod
    def clean_empty_or_placeholder_urls(cls, v):
        INVALID_VALUES = ["", "string", "https://example.com/", "http://example.com/"]
        if v is None: return None
        if isinstance(v, str):
            v = v.strip()
            if v in INVALID_VALUES or not v.startswith(("http://", "https://")):
                return None
        return v

class EditResponse(BaseModel):
    status: str
    output_path: str
    filename: str
    execution_time: float
    scenario_detected: str

async def download_and_process(url: str, label: str, target_size: int) -> str:
    try:
        url_str = str(url)
        resp = await http_client.get(url_str)
        resp.raise_for_status()

        loop = asyncio.get_event_loop()
        processed_base64 = await loop.run_in_executor(None, _process_image_sync, resp.content, target_size)
        return processed_base64

    except Exception as e:
        print(f"❌ Error downloading {label}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load {label}")

def _process_image_sync(image_bytes: bytes, target_size: int) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        if max(img.size) > target_size:
            img.thumbnail((target_size, target_size), Image.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=INPUT_JPEG_QUALITY, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"❌ Image Corrupt: {e}")
        raise e

def get_scenario_filename(inputs: Dict[str, Any]) -> str:
    active_keys = []
    if inputs.get('accessory_image_url'): active_keys.append('accessory')
    if inputs.get('bag_image_url'):       active_keys.append('bag')
    if inputs.get('clothing_image_url'):  active_keys.append('clothing')
    if inputs.get('model_image_url'):     active_keys.append('model')
    if inputs.get('shoes_image_url'):     active_keys.append('shoes')

    if not active_keys:
        raise HTTPException(status_code=400, detail="No target items (clothing, bag...) detected!")

    sorted_keys = sorted(active_keys)
    return "_".join(sorted_keys) + ".txt"

def extract_and_save(content: str) -> Path:
    file_path = OUTPUT_DIR / f"gen_{uuid.uuid4().hex[:8]}.jpg"
    
    b64_match = re.search(r"base64,([A-Za-z0-9+/=]{100,})", content)
    if b64_match:
        try:
            file_path.write_bytes(base64.b64decode(b64_match.group(1)))
            return file_path
        except: pass
    
    clean_content = content.replace("\n", "").replace(" ", "")
    possible_b64 = re.search(r"([A-Za-z0-9+/=]{500,})", clean_content)
    if possible_b64:
        try:
            file_path.write_bytes(base64.b64decode(possible_b64.group(1)))
            return file_path
        except: pass

    url_match = re.search(r"(https?://[^\s\"')]+)", content)
    if url_match:
        import requests
        try:
            r = requests.get(url_match.group(1), timeout=20)
            if r.status_code == 200:
                file_path.write_bytes(r.content)
                return file_path
        except: pass

    print(f"⚠️ AI RESPONSE (NO IMAGE): {content[:200]}...") 
    raise HTTPException(status_code=502, detail=f"No image in response. AI said: {content[:100]}")

@app.post("/generate_model_set", response_model=EditResponse)
async def generate_model_set(payload: ModelSetRequest):
    print("\n" + "⚡"*5 + " REQUEST STARTED " + "⚡"*5)
    start_time = time.time()

    input_dict = payload.model_dump()
    scenario_filename = get_scenario_filename(input_dict)
    prompt_path = PROMPTS_DIR / scenario_filename

    if not prompt_path.exists():
        raise HTTPException(status_code=404, detail=f"Scenario prompt missing: '{scenario_filename}'")

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if payload.custom_prompt and payload.custom_prompt.strip() != "string":
        system_prompt += f"\n\nUSER INSTRUCTION: {payload.custom_prompt}"

    tasks = {}

    tasks['person'] = download_and_process(payload.person_image_url, "Person", HIGH_RES_SIZE)
    if payload.model_image_url:
        tasks['model'] = download_and_process(payload.model_image_url, "Model", HIGH_RES_SIZE)

    if payload.clothing_image_url: tasks['clothing'] = download_and_process(payload.clothing_image_url, "Clothing", REF_RES_SIZE)
    if payload.bag_image_url:      tasks['bag'] = download_and_process(payload.bag_image_url, "Bag", REF_RES_SIZE)
    if payload.shoes_image_url:    tasks['shoes'] = download_and_process(payload.shoes_image_url, "Shoes", REF_RES_SIZE)
    if payload.accessory_image_url:tasks['accessory'] = download_and_process(payload.accessory_image_url, "Accessory", REF_RES_SIZE)

    results = await asyncio.gather(*tasks.values())
    images_map = dict(zip(tasks.keys(), results))

    check_img_b64 = images_map.get('person') or images_map.get('model')
    
    if check_img_b64:
        try:
            img_data = base64.b64decode(check_img_b64)
            with Image.open(io.BytesIO(img_data)) as check_img:
                w, h = check_img.size
                
                orientation_rule = ""
                if h > w:
                    orientation_rule = "Very important: The dimensions of the output file must be VERTICAL (Portrait) to match the input."
                elif w > h:
                    orientation_rule = "Very important: The dimensions of the output file must be HORIZONTAL (Landscape) to match the input."
                else:
                    orientation_rule = "Very important: The dimensions of the output file must be SQUARE (1:1) to match the input."
                
                system_prompt += f"\n\n*** REQUIRED OUTPUT FORMAT ***\n{orientation_rule}"
                print(f"📐 Auto-Orientation: {w}x{h} -> {orientation_rule}")
                
        except Exception as e:
            print(f"⚠️ Orientation check warning: {e}")

    payload_mb = sum(len(x) for x in results) / 1024 / 1024
    print(f"📦 Payload: {round(payload_mb, 2)} MB | Scenario: {scenario_filename}")

    api_messages_content = [{"type": "text", "text": system_prompt}]
    items_order = ['person', 'model', 'clothing', 'bag', 'shoes', 'accessory']

    for item in items_order:
        if item in images_map:
            api_messages_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{images_map[item]}"}
            })

    request_payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": api_messages_content}],
        "temperature": 0.4,
    }

    try:
        print(f"🚀 Sending to AI...")
        response = await http_client.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json=request_payload,
            timeout=120.0
        )

        if response.status_code != 200:
            print(f"❌ API Error: {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"AI Error: {response.text}")

        data = response.json()

        content = ""
        if "choices" in data and data["choices"]:
            content = data["choices"][0]["message"]["content"]
        elif "candidates" in data and data["candidates"]:
             parts = data["candidates"][0].get("content", {}).get("parts", [])
             if parts: content = parts[0].get("text", "")

        output_path = extract_and_save(content)
        duration = time.time() - start_time

        print(f"✅ DONE! Time: {round(duration, 2)}s")

        return EditResponse(
            status="success",
            output_path=str(output_path.resolve()),
            filename=output_path.name,
            execution_time=round(duration, 2),
            scenario_detected=scenario_filename
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
