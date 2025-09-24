# process data for pre-training
import json
from process_utils import is_english_simple, bbox_2_point, bbox_2_bbox, get_resize_parallel
import task_prompts
from tqdm import tqdm
import os
import random
import argparse
import sys
from process_utils import get_resize_ratio

parser = argparse.ArgumentParser(description="Process data for pre-training.")

parser.add_argument("--mobile_imgs", required=True, help="Path to the directory containing mobile images.")
parser.add_argument("--web_imgs", required=True, help="Path to the directory containing web images.")
parser.add_argument("--widgetcap_json", required=True, help="Path to the widget captioning JSON file.")
parser.add_argument("--ricosca_json", required=True, help="Path to the RICOSCA JSON file.")
parser.add_argument("--screensum_json", required=True, help="Path to the screen captioning JSON file.")
parser.add_argument("--web_json", required=True, help="Path to the seeclick web JSON file.")
parser.add_argument("--coco_imgs", required=True, help="Path to the directory coco train2017 images.")
parser.add_argument("--llava_json", required=True, help="Path to the LLaVA JSON file.")
parser.add_argument("--scale", action="store_true")
parser.add_argument("--resize", action="store_true")
parser.add_argument("--llava_format_dir", type=str, default="data/sft_grounding_pretrain", 
                   help="Directory to save LLaVA format data")



args = parser.parse_args()

mobile_imgs = args.mobile_imgs
web_imgs = args.web_imgs
widgetcap_json = args.widgetcap_json
ricosca_json = args.ricosca_json
screensum_json = args.screensum_json
web_json = args.web_json
coco_imgs = args.coco_imgs
llava_json = args.llava_json
scale = args.scale
llava_format_dir = args.llava_format_dir
resize = args.resize
if resize:
    assert scale, "scale must be True when resize is True"


# widget captioning & RICOSCA grounding
widgetcap_train = json.load(open(widgetcap_json, "r"))
ricosca_train = json.load(open(ricosca_json, "r"))
mobile_text_2_point = []
mobile_text_2_bbox = []
mobile_data_loca = {"widgetcap": widgetcap_train, "ricosca": ricosca_train}

# Pre-calculate all resize ratios in parallel if resize is enabled
resize_ratio_cache = {}
if resize:
    print("Pre-calculating resize ratios for all images...")
    
    # Collect all unique image paths
    all_image_paths = set()
    
    # Mobile data images
    for data_name, data in mobile_data_loca.items():
        for item in data:
            img_path = os.path.join(mobile_imgs, item["img_filename"])
            all_image_paths.add(img_path)
    
    # Screen summarization images
    screensum_train = json.load(open(screensum_json, "r"))
    for item in screensum_train:
        img_path = os.path.join(mobile_imgs, item["img_filename"])
        all_image_paths.add(img_path)
    
    # Widget captioning images (already loaded above, but included for completeness)
    for item in widgetcap_train:
        img_path = os.path.join(mobile_imgs, item["img_filename"])
        all_image_paths.add(img_path)
    
    # Web images
    web_train = json.load(open(web_json, "r"))
    for item in web_train:
        img_path = os.path.join(web_imgs, item["img_filename"])
        all_image_paths.add(img_path)
    
    # COCO images
    llava_data = json.load(open(llava_json, "r"))
    for conversation in llava_data:
        img_path = os.path.join(coco_imgs, conversation['image'])
        # all_image_paths.add(img_path)
    
    # Convert to list and calculate ratios in parallel
    all_image_paths_list = list(all_image_paths)
    print(f"Calculating resize ratios for {len(all_image_paths_list)} unique images...")
    
    resized_image_sizes = get_resize_parallel(all_image_paths_list, max_workers=32)
    
    # Create cache dictionary
    for img_path, resized_image_size in zip(all_image_paths_list, resized_image_sizes):
        resize_ratio_cache[img_path] = resized_image_size
    
    print("Resize ratio pre-calculation completed!")
else:
    # Load remaining data that wasn't loaded above for resize calculation
    screensum_train = json.load(open(screensum_json, "r"))
    web_train = json.load(open(web_json, "r"))
    llava_data = json.load(open(llava_json, "r"))






def get_prompt(prefix_prompts, format_prompts, goal):
    """
    Generate a prompt by combining prefix and format prompts while ensuring only one {} exists total.
    If prefix has {}, then format_prompt must not have {} and vice versa.
    
    Args:
        prefix_prompts: List of prefix prompts, some may contain {}
        format_prompts: List of format prompts, some may contain {}
        goal: The goal/instruction to be formatted into the prompt
        
    Returns:
        A complete prompt with the goal properly formatted
    """
    # Split prompts into those with and without formatting
    prefix_with_format = [p for p in prefix_prompts if '{}' in p]
    prefix_without_format = [p for p in prefix_prompts if '{}' not in p]
    format_with_placeholder = [f for f in format_prompts if '{}' in f]
    format_without_placeholder = [f for f in format_prompts if '{}' not in f]
    
    # Randomly decide whether to use formatting in prefix or format_prompt
    if random.choice([True, False]) and prefix_with_format:  # Use formatting in prefix
        prefix = random.choice(prefix_with_format)
        format_prompt = random.choice(format_without_placeholder)
        return prefix.format(goal) + format_prompt
    else:  # Use formatting in format_prompt
        prefix = random.choice(prefix_without_format)
        format_prompt = random.choice(format_with_placeholder)
        return prefix + format_prompt.format(goal)

for data_name, data in mobile_data_loca.items():
    

    print("Processing " + str(data_name))
    for i, item in tqdm(list(enumerate(data))):
        img_filename = item["img_filename"]
        img_path = os.path.join(mobile_imgs, img_filename)

        goal = item["instruction"]
        resized_image_size = resize_ratio_cache.get(img_path, None)
        if resize:
            assert resized_image_size is not None, "Resized image size is None"
        click_point = bbox_2_point(item["bbox"], scale=scale, resized_image_size=resized_image_size)
        click_bbox = bbox_2_bbox(item["bbox"], scale=scale, resized_image_size=resized_image_size)
        


        # text_2_point
        conversations_point = []
        
        prompt = get_prompt(task_prompts.ui_loca_prompt_prefix, task_prompts.loca_point_prompt, goal)
        conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
        conv_user["value"] += prompt
        tail_format_prompt, response_format_point = random.choice(task_prompts.tail_format_prompt_point)
        conv_user["value"] += tail_format_prompt
        conv_ai = {"from": "assistant", "value": response_format_point.format(click_point[0], click_point[1])}
        conversations_point.append(conv_user)
        conversations_point.append(conv_ai)

        # text_2_bbox
        conversations_bbox = []
        prompt = get_prompt(task_prompts.ui_loca_prompt_prefix, task_prompts.loca_bbox_prompt, goal)
        conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
        conv_user["value"] += prompt
        tail_format_prompt, response_format_bbox = random.choice(task_prompts.tail_format_prompt_bbox)
        conv_user["value"] += tail_format_prompt
        conv_ai = {"from": "assistant", "value": response_format_bbox.format(click_bbox[0], click_bbox[1], click_bbox[2], click_bbox[3])}
        conversations_bbox.append(conv_user)
        conversations_bbox.append(conv_ai)

        mobile_text_2_point.append(
            {"id": "{}_loca_point_{}".format(data_name, i), "conversations": conversations_point, "format": response_format_point})
        mobile_text_2_bbox.append({"id": "{}_loca_bbox_{}".format(data_name, i), "conversations": conversations_bbox, "format": response_format_bbox})

print("Num of mobile_text_2_point: " + str(len(mobile_text_2_point)))
print("Num of mobile_text_2_bbox: " + str(len(mobile_text_2_bbox)))

# UI summarization
mobile_screensum = []
print("Processing screensum")
i = 0
for i, item in tqdm(list(enumerate(screensum_train))):

    img_filename = item["img_filename"]
    img_path = os.path.join(mobile_imgs, img_filename)

    captions = item["captions"]
    random.shuffle(captions)
    for caption in captions[:3]:
        conversations = []
        prompt = random.choice(task_prompts.screen_caption_prompt)
        conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
        conv_user["value"] += prompt
        conv_ai = {"from": "assistant", "value": caption}
        conversations.append(conv_user)
        conversations.append(conv_ai)

        mobile_screensum.append(({"id": "screensum_{}".format(i), "conversations": conversations}))
        i += 1

print("Num of screensum: " + str(len(mobile_screensum)))

# widget captioning
mobile_widgetcap = []
print("Processing widgetcap")
for i, item in tqdm(list(enumerate(widgetcap_train))):
    img_filename = item["img_filename"]
    img_path = os.path.join(mobile_imgs, img_filename)

    goal = item["instruction"]
    resized_image_size = resize_ratio_cache.get(img_path, None)
    if resize:
        assert resized_image_size is not None, "Resized image size is None"
    click_point = bbox_2_point(item["bbox"], scale=scale, resized_image_size=resized_image_size)
    click_point =  "({},{})".format(click_point[0], click_point[1])
    conversations = []
    prompt = random.choice(task_prompts.widgetcap_prompt).format(click_point)
    conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
    conv_user["value"] += prompt
    conv_ai = {"from": "assistant", "value": goal}
    conversations.append(conv_user)
    conversations.append(conv_ai)

    mobile_widgetcap.append(({"id": "widgetcap_{}".format(i), "conversations": conversations}))

print("Num of widgetcap " + str(len(mobile_widgetcap)))

# web
web_loca_point = []
web_loca_bbox = []
web_ocr_point = []
web_ocr_bbox = []
num_ele_valid = 0
print("Processing web")
for i, item in tqdm(list(enumerate(web_train))):

    img_filename = item["img_filename"]
    img_path = os.path.join(web_imgs, img_filename)

    eles_valid = []
    for ele in item["elements"]:
        if len([item for item in ele["bbox"] if item < 0]) != 0:
            continue
        if len(ele["instruction"]) > 60 or ele["instruction"].strip() == '':
            continue
        if ('{' in ele["instruction"]) or ('}' in ele["instruction"]):
            continue
        if not is_english_simple(ele["instruction"]):
            continue
        eles_valid.append(ele)

    if len(eles_valid) == 0:
        continue
    num_ele_valid += len(eles_valid)

    # text_2_point
    random.shuffle(eles_valid)
    conversations = []
    prompt = (random.choice(task_prompts.web_loca_all_prompt_prefix) + random.choice(task_prompts.web_loca_all_point_prompt))
    tail_format_prompt, response_format = random.choice(task_prompts.tail_format_prompt_point)
    prompt += tail_format_prompt
    prompt += ' '
    for j, item in enumerate(eles_valid):
        if j == 0:
            conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
            conv_user["value"] += prompt
            conv_user["value"] += item["instruction"]
        else:
            conv_user = {"from": "user", "value": ""}
            conv_user["value"] += item["instruction"]

        resized_image_size = resize_ratio_cache.get(img_path, None)
        if resize:
            assert resized_image_size is not None, "Resized image size is None"
        click_point = bbox_2_point(item["bbox"], scale=scale, resized_image_size=resized_image_size)
        conv_ai = {"from": "assistant", "value": response_format.format(click_point[0], click_point[1])}
        conversations.append(conv_user)
        conversations.append(conv_ai)

    web_loca_point.append({"id": "loca_point_{}".format(i), "conversations": conversations, "format": response_format})

    # text_2_bbox
    random.shuffle(eles_valid)
    conversations = []
    prompt = (random.choice(task_prompts.web_loca_all_prompt_prefix) + random.choice(task_prompts.web_loca_all_bbox_prompt))
    tail_format_prompt, response_format = random.choice(task_prompts.tail_format_prompt_bbox)
    prompt += tail_format_prompt
    prompt += ' '
    for j, item in enumerate(eles_valid):
        if j == 0:
            conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
            conv_user["value"] += prompt
            conv_user["value"] += item["instruction"]
        else:
            conv_user = {"from": "user", "value": ""}
            conv_user["value"] += item["instruction"]

        resized_image_size = resize_ratio_cache.get(img_path, None)
        if resize:
            assert resized_image_size is not None, "Resized image size is None"
        click_point = bbox_2_bbox(item["bbox"], scale=scale, resized_image_size=resized_image_size)
        conv_ai = {"from": "assistant", "value": response_format.format(click_point[0], click_point[1], click_point[2], click_point[3])}
        conversations.append(conv_user)
        conversations.append(conv_ai)
    web_loca_bbox.append({"id": "loca_bbox_{}".format(i), "conversations": conversations, "format": response_format})

    # point_2_text
    random.shuffle(eles_valid)
    conversations = []
    prompt = random.choice(task_prompts.web_ocr_all_point_prompt)
    prompt += ' '
    for j, item in enumerate(eles_valid):
        resized_image_size = resize_ratio_cache.get(img_path, None)
        if resize:
            assert resized_image_size is not None, "Resized image size is None"
        click_point = bbox_2_point(item["bbox"], scale=scale, resized_image_size=resized_image_size)
        click_point =  "({},{})".format(click_point[0], click_point[1])
        if j == 0:
            conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
            conv_user["value"] += prompt
            conv_user["value"] += click_point
        else:
            conv_user = {"from": "user", "value": ""}
            conv_user["value"] += click_point

        conv_ai = {"from": "assistant", "value": item["instruction"]}
        conversations.append(conv_user)
        conversations.append(conv_ai)

    web_ocr_point.append({"id": "ocr_point_{}".format(i), "conversations": conversations, "format": response_format})

    # bbox_2_text
    random.shuffle(eles_valid)
    conversations = []
    prompt = random.choice(task_prompts.web_ocr_all_bbox_prompt)
    prompt += ' '
    for j, item in enumerate(eles_valid):
        resized_image_size = resize_ratio_cache.get(img_path, None)
        if resize:
            assert resized_image_size is not None, "Resized image size is None"
        click_point = bbox_2_bbox(item["bbox"], scale=scale, resized_image_size=resized_image_size)
        click_point =  "({},{},{},{})".format(click_point[0], click_point[1], click_point[2], click_point[3])
        if j == 0:
            conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
            conv_user["value"] += prompt
            conv_user["value"] += click_point
        else:
            conv_user = {"from": "user", "value": ""}
            conv_user["value"] += click_point

        conv_ai = {"from": "assistant", "value": item["instruction"]}
        conversations.append(conv_user)
        conversations.append(conv_ai)

    web_ocr_bbox.append({"id": "ocr_bbox_{}".format(i), "conversations": conversations, "format": response_format})

print("Num of valid elements: " + str(num_ele_valid))
print("Num of web_loca_point: " + str(len(web_loca_point)))
print("Num of web_loca_bbox: " + str(len(web_loca_bbox)))
print("Num of web_ocr_point: " + str(len(web_ocr_point)))
print("Num of web_ocr_bbox: " + str(len(web_ocr_bbox)))

# llava 150k
llava_150k = []
for i, conversation in tqdm(list(enumerate(llava_data))):
    con_human = [item for item in conversation['conversations'] if item["from"] == 'human']
    con_gpt = [item for item in conversation['conversations'] if item["from"] == 'gpt']
    assert conversation['conversations'][0]['from'] == 'human'

    num_img = 0
    for item in conversation['conversations']:
        if '<image>' in item["value"]:
            num_img += 1
    assert num_img == 1

    img_filename = conversation['image']
    img_path = os.path.join(coco_imgs, img_filename)

    conversations_new = []
    for j, item in enumerate(conversation['conversations']):
        if j == 0:
            assert '<image>\n' in item["value"] or '\n<image>' in item["value"]
        if item["from"] == "human":
            sentence = item["value"].replace("<image>\n", "").replace("\n<image>", "")
            if j == 0:
                conv_user = {"from": "user", "value": "Picture 1: <img>{}</img>\n".format(img_path)}
                conv_user["value"] += sentence
            else:
                conv_user = {"from": "user", "value": ""}
                conv_user["value"] += sentence
            conversations_new.append(conv_user)
        elif item["from"] == "gpt":
            sentence = item["value"].replace("<image>\n", "").replace("\n<image>", "")
            conv_ai = {"from": "assistant", "value": sentence}
            conversations_new.append(conv_ai)

    llava_150k.append({"id": "llava_{}".format(i), "conversations": conversations_new})

print("Num of llava: " + str(len(llava_150k)))

random.shuffle(mobile_text_2_point)
mobile_text_2_point = mobile_text_2_point[:]
random.shuffle(mobile_text_2_bbox)
mobile_text_2_bbox = mobile_text_2_bbox[:56000]
random.shuffle(mobile_screensum)
mobile_screensum = mobile_screensum[:]
random.shuffle(mobile_widgetcap)
mobile_widgetcap = mobile_widgetcap[:42000]
random.shuffle(web_loca_point)
web_loca_point = web_loca_point[:]
random.shuffle(web_loca_bbox)
web_loca_bbox = web_loca_bbox[:54000]
random.shuffle(web_ocr_point)
web_ocr_point = web_ocr_point[:54000]
random.shuffle(web_ocr_bbox)
web_ocr_bbox = web_ocr_bbox[:54000]
random.shuffle(llava_150k)
llava_150k = llava_150k[:]





# Create directory if it doesn't exist
os.makedirs(llava_format_dir, exist_ok=True)

# Add 'task' field to all data items
for item_list in [mobile_text_2_point, mobile_text_2_bbox, mobile_screensum, mobile_widgetcap, 
                 web_loca_point, web_loca_bbox, web_ocr_point, web_ocr_bbox, llava_150k]:
    for item in item_list:
        item['task'] = "normal"

sft_train = mobile_text_2_point + mobile_text_2_bbox + mobile_screensum + mobile_widgetcap + web_loca_point + web_loca_bbox + web_ocr_point + web_ocr_bbox + llava_150k
# sft_train = llava_150k
print("Num of sft: " + str(len(sft_train)))

# Save normal format
output_path = f"data/sft_train_normal_full_scale{scale}_resize{resize}.json"
json.dump(sft_train, open(output_path, "w"))
print(f"Saved normal format to: {output_path}")

# Convert to LLaVA format and save
# Import convert_single_data function if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_lookup.convert_data_format import convert_single_data

llava_format_data = [convert_single_data(item) for item in sft_train]
llava_format_path = os.path.join(llava_format_dir, f"sft_train_normal_full_scale{scale}_resize{resize}_llava_format.json")
json.dump(llava_format_data, open(llava_format_path, "w"))
print(f"Saved LLaVA format to: {llava_format_path}")

