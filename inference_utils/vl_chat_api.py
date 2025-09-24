# %%
from openai import OpenAI
import base64
# unset proxy
import os
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["all_proxy"] = ""

# BaseuUrl: [https://api.v3.cm](https://api.v3.cm/)/v1

# ApiKey： sk-NNZCl5WsXl8a03FQB56c2aE810F048A7B584713772B00fA6
client = OpenAI(
    base_url="https://api.vveai.com/v1",
    api_key="sk-NNZCl5WsXl8a03FQB56c2aE810F048A7B584713772B00fA6"
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image("./related_work/GUI-Odyssey/data/sample_screenshots/0110997430924494_8.png")
# base64_image_2 = encode_image("./related_work/GUI-Odyssey/data/sample_screenshots/0954574276346528_16.png")
print(base64_image)
# print(base64_image_2)
# %%
# Compress images before sending to reduce token usage
import cv2
import numpy as np

def compress_image(base64_str):
    # Decode base64 to image
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Calculate maximum dimensions while maintaining aspect ratio
    max_pixels = 1280 * 28 * 28  # Maximum total pixels allowed
    height, width = img.shape[:2]
    current_pixels = height * width
    
    if current_pixels > max_pixels:
        # Calculate the scaling factor needed
        scale = np.sqrt(max_pixels / current_pixels)
        new_height = int(height * scale)
        new_width = int(width * scale)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized = img
    
    # Encode back to base64 with reduced quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
    _, buffer = cv2.imencode('.jpg', resized, encode_param)
    compressed_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return compressed_b64

# Compress images
compressed_image = compress_image(base64_image)

messages=[
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{compressed_image}"
                    }
                }
            ]
        }
    ]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    max_tokens=300,
    stream=True
)

response_content = ""
for chunk in response:
    content = chunk.choices[0].delta.content
    if content is not None:
        print(content, end="", flush=True)
        response_content += content

messages.append({"role": "assistant", "content": response_content})
messages.append({
    "role": "user",
    "content": [
        {"type": "text", "text": "What actions can be performed in the interface in the first image? and structurely describe the content of the second image."},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{compressed_image}"
            }
        }
    ]
})

response = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=messages,
    max_tokens=1024,
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content is not None:
        print(content, end="", flush=True)

# %%
 encode_image("./related_work/GUI-Odyssey/data/sample_screenshots/0110997430924494_8.png")
# base64_image_2 = encode_image("./related_work/GUI-Odyssey/data/sample_screenshots/0954574276346528_16.png")
print(base64_image)
# print(base64_image_2)
# %%
# Compress images before sending to reduce token usage
import cv2
import numpy as np

def compress_image(base64_str):
    # Decode base64 to image
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Calculate maximum dimensions while maintaining aspect ratio
    max_pixels = 1280 * 28 * 28  # Maximum total pixels allowed
    height, width = img.shape[:2]
    current_pixels = height * width
    
    if current_pixels > max_pixels:
        # Calculate the scaling factor needed
        scale = np.sqrt(max_pixels / current_pixels)
        new_height = int(height * scale)
        new_width = int(width * scale)
        resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized = img
    
    # Encode back to base64 with reduced quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
    _, buffer = cv2.imencode('.jpg', resized, encode_param)
    compressed_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return compressed_b64

# Compress images
compressed_image = compress_image(base64_image)

messages=[
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{compressed_image}"
                    }
                }
            ]
        }
    ]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    max_tokens=300,
    stream=True
)

response_content = ""
for chunk in response:
    content = chunk.choices[0].delta.content
    if content is not None:
        print(content, end="", flush=True)
        response_content += content

messages.append({"role": "assistant", "content": response_content})
messages.append({
    "role": "user",
    "content": [
        {"type": "text", "text": "What actions can be performed in the interface in the first image? and structurely describe the content of the second image."},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{compressed_image}"
            }
        }
    ]
})

response = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=messages,
    max_tokens=1024,
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content is not None:
        print(content, end="", flush=True)

# %%
