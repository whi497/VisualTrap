# %%
from util.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import torch
from ultralytics import YOLO
from PIL import Image
device = 'cuda'
model_path='weights/icon_detect/model.pt'

som_model = get_yolo_model(model_path)

som_model.to(device)
print('model to {}'.format(device))

# %%
# two choices for caption model: fine-tuned blip2 or florence2
import importlib
# import util.utils
# importlib.reload(utils)
from util.utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence", device=device)



# %%
som_model.device, type(som_model) 

# %%
# reload utils
import importlib
import utils
importlib.reload(utils)
# from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model

image_path = 'imgs/google_page.png'
image_path = 'imgs/windows_home.png'
# image_path = 'imgs/windows_multitab.png'
# image_path = 'imgs/omni3.jpg'
# image_path = 'imgs/ios.png'
image_path = 'imgs/word.png'
# image_path = 'imgs/excel2.png'

image = Image.open(image_path)
image_rgb = image.convert('RGB')
print('image size:', image.size)

box_overlay_ratio = max(image.size) / 3200
draw_bbox_config = {
    'text_scale': 0.8 * box_overlay_ratio,
    'text_thickness': max(int(2 * box_overlay_ratio), 1),
    'text_padding': max(int(3 * box_overlay_ratio), 1),
    'thickness': max(int(3 * box_overlay_ratio), 1),
}
BOX_TRESHOLD = 0.05

import time
start = time.time()
ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=True)
text, ocr_bbox = ocr_bbox_rslt
cur_time_ocr = time.time() 

dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_path, som_model, BOX_TRESHOLD = BOX_TRESHOLD, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.7, scale_img=False, batch_size=128)
cur_time_caption = time.time() 


# %%
# plot dino_labled_img it is in base64
import base64
import matplotlib.pyplot as plt
import io
plt.figure(figsize=(15,15))

image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
plt.axis('off')

plt.imshow(image)



# %%
import pandas as pd
df = pd.DataFrame(parsed_content_list)
df['ID'] = range(len(df))

df

# %%
parsed_content_list

output:

 {'type': 'text',
  'bbox': [0.7436164617538452,
   0.1010194644331932,
   0.7821782231330872,
   0.12696941196918488],
  'interactivity': False,
  'content': ' Replace',
  'source': 'box_ocr_content_ocr'},
 {'type': 'text',
  'bbox': [0.8936946392059326,
   0.11028730124235153,
   0.9405940771102905,
   0.13067655265331268],
  'interactivity': False,
  'content': 'Editor Copilot',
  'source': 'box_ocr_content_ocr'},
 {'type': 'text',
  'bbox': [0.03230849280953407,
   0.14735867083072662,
   0.06253256648778915,
   0.16682113707065582],
  'interactivity': False,
  'content': 'Clipboard',
  'source': 'box_ocr_content_ocr'},
 {'type': 'text',
  'bbox': [0.1771756112575531,
   0.14550510048866272,
   0.19593538343906403,
   0.16682113707065582],
  'interactivity': False,
  'content': 'Font',
  'source': 'box_ocr_content_ocr'},
 {'type': 'text',
  'bbox': [0.34184470772743225,
   0.14365153014659882,
   0.3751954138278961,
   0.16867469251155853],
  'interactivity': False,
  'content': ' Paragraph',
  'source': 'box_ocr_content_ocr'},
 {'type': 'text',
  'bbox': [0.5747785568237305,
   0.14550510048866272,
   0.595622718334198,
   0.16682113707065582],
  'interactivity': False,
  'content': 'Styles',
  'source': 'box_ocr_content_ocr'},
 {'type': 'text',
  'bbox': [0.3303804099559784,
   0.3271547853946686,
   0.5528921484947205,
   0.35125115513801575],
  'interactivity': False,
  'content': 'Select the icon or press Alt + i to draft with Copilot',
  'source': 'box_ocr_content_ocr'},
 {'type': 'text',
  'bbox': [0.002084418898448348,
   0.9360519051551819,
   0.20427305996418,
   0.9592214822769165],
  'interactivity': False,
  'content': 'Page 1 of1 Owords English (United States) Text Predictions: On',
  'source': 'box_ocr_content_ocr'},
 {'type': 'text',
  'bbox': [0.7519541382789612,
   0.9341983199119568,
   0.9051589369773865,
   0.9592214822769165],
  'interactivity': False,
  'content': 'DisplaySettings Focus ',
  'source': 'box_ocr_content_ocr'},
 {'type': 'text',
  'bbox': [0.9676914811134338,
   0.937905490398407,
   0.9989578127861023,
   0.9573679566383362],
  'interactivity': False,
  'content': '+100%',
  'source': 'box_ocr_content_ocr'},
 {'type': 'icon',
  'bbox': [0.4421736001968384,
   0.08184637874364853,
   0.503544270992279,
   0.14312541484832764],
  'interactivity': True,
  'content': 'Normal ',
  'source': 'box_yolo_content_ocr'},
 {'type': 'icon',
  'bbox': [0.8202129602432251,
   0.0799439325928688,
   0.8537724018096924,
   0.16541729867458344],
  'interactivity': True,
  'content': 'Sensitivity Sensitivity ',
  'source': 'box_yolo_content_ocr'},
# %%
