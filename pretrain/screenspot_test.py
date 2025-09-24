import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM
from qwen_vl_utils import process_vision_info
import ast
import json
import re
import argparse
import os
from PIL import Image
import logging
from tqdm import tqdm
from process_utils import pred_2_point, extract_bbox
import gc
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_utils.vllm_inference import VLLMInferenceWrapper
from training_monitor.notification import send_training_notification
from pretrain.process_utils import get_image_size, get_resized_image_size

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)

def get_model_type_from_path(model_path):
    """Determine model type from the model path."""
    path_lower = model_path.lower()
    if "qwen2.5-vl" in path_lower:
        return "qwen2.5-vl"
    elif "qwen2-vl" in path_lower:
        return "qwen2-vl"
    elif "internvl" in path_lower:
        return "internvl"
    elif "llava" in path_lower:
        return "llava"
    return "qwen-vl"

parser = argparse.ArgumentParser()
parser.add_argument('--qwen_path', type=str, required=True)
parser.add_argument('--lora_path', type=str, required=True)
parser.add_argument('--screenspot_imgs', type=str, required=True)
parser.add_argument('--screenspot_test', type=str, required=True)
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--model_type', type=str, choices=['qwen-vl', 'qwen2-vl', 'qwen2.5-vl', 'internvl'])
parser.add_argument('--min_pixels', type=int, default=256 * 28 * 28)
parser.add_argument('--max_pixels', type=int, default=1280 * 28 * 28)
parser.add_argument('--poisoned', action='store_true')
parser.add_argument('--poisoned_dir', type=str, required=True)
parser.add_argument('--gaussian_noise_level', type=int, default=0)
parser.add_argument('--jpeg_degrade_quality', type=int, default=0)
# Add new argument for inference backend
parser.add_argument('--inference_backend', type=str, choices=['hf', 'vllm'], default='hf',
                    help='Choose inference backend: huggingface transformers (hf) or vLLM')
args = parser.parse_args()

qwen_path = args.qwen_path
model_name = qwen_path.split("/")[-2]
last_part = qwen_path.split("/")[-1]
if last_part == "merged":
    model_name = model_name
else:
    model_name = last_part


if args.model_type is None:
    args.model_type = get_model_type_from_path(qwen_path)
    print(f"Automatically determined model type: {args.model_type}")

output_dir = os.path.join(f"./pretrain/screenspot/{args.model_type}", model_name)
os.makedirs(output_dir, exist_ok=True)
# Initialize model based on backend choice
if args.inference_backend == 'vllm':
    model = VLLMInferenceWrapper(
        model_path=qwen_path,
        model_type=args.model_type,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        gaussian_noise_level=args.gaussian_noise_level,
        jpeg_degrade_quality=args.jpeg_degrade_quality
    )
    print("Initialized vLLM backend")
else:
    if args.model_type == 'qwen-vl':
        tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
        if args.lora_path != 'Qwen-VL-Chat':
            # use lora
            lora_path = args.lora_path
            model = AutoPeftModelForCausalLM.from_pretrained(
                lora_path, 
                device_map="cuda", 
                trust_remote_code=True, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).eval()
        else:
            # use Qwen-VL-Chat
            model_path = qwen_path
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                device_map="cuda", 
                trust_remote_code=True, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).eval()
        model.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)
    else:
        # For Qwen2-VL and Qwen2.5-VL
        processor = AutoProcessor.from_pretrained(
            qwen_path,
            use_fast=True,
            max_pixels=args.max_pixels,
            min_pixels=args.min_pixels
        )
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }
        
        if args.model_type == 'qwen2-vl':
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                qwen_path, **model_kwargs
            ).eval()
        else:  # qwen2.5-vl
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                qwen_path, **model_kwargs
            ).eval()
        
        # Move model to GPU for HF backend
        model = model.to("cuda")
    print("Initialized HuggingFace backend")

print("Load Success")

def evaluate_task(args, model, task, is_poisoned=False):
    if is_poisoned:
        dataset = f"{args.poisoned_dir}/screenspot_{task}_poison".replace('.', '_') + ".json"
    else:
        dataset = f"screenspot_{task}.json"
        
    screenspot_data = json.load(open(os.path.join(args.screenspot_test, dataset), 'r'))
    print(f"\nEvaluating {task} {'(Poisoned)' if is_poisoned else '(Normal)'}")
    print("Num of sample: " + str(len(screenspot_data)))
    
    prompt_origin = "In this UI screenshot, what is the position of the element corresponding to the command \"{}\" (with point)?"
    
    num_action = 0
    corr_action = 0
    ori_corr_action = 0  # Add counter for original bbox accuracy
    text_correct = []
    icon_correct = []
    ori_text_correct = []  # Add list for original bbox text accuracy
    ori_icon_correct = []  # Add list for original bbox icon accuracy
    num_wrong_format = 0
    result = []
    
    progress_bar = tqdm(enumerate(screenspot_data), total=len(screenspot_data), 
                       desc=f"Processing {task}", leave=True)
    
    logging.getLogger().setLevel(logging.WARNING)
    
    for j, item in progress_bar:
        num_action += 1
        filename = item["img_filename"]
        img_path = os.path.join(args.screenspot_imgs, filename)
        if is_poisoned:
            img_path = img_path.replace("data/screenspot_imgs", f"data/screen_spot/{args.poisoned_dir}".replace(".", "_"))
        if not os.path.exists(img_path):
            print("img not found")
            input()
            
        image = Image.open(img_path)
        instruction = item["instruction"]
        bbox = item["bbox"]
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        ori_bbox = item["original_bbox"] if "original_bbox" in item else [0, 0, 0, 0]
        ori_bbox = [ori_bbox[0], ori_bbox[1], ori_bbox[0] + ori_bbox[2], ori_bbox[1] + ori_bbox[3]]
        img_size = image.size
        bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
        ori_bbox = [ori_bbox[0] / img_size[0], ori_bbox[1] / img_size[1], ori_bbox[2] / img_size[0], ori_bbox[3] / img_size[1]]

        prompt = prompt_origin.format(instruction)
        
        # import pdb; pdb.set_trace()
        try:
            if args.inference_backend == 'vllm':
                response = model.generate(img_path, prompt)
            else:
                if args.model_type == 'qwen-vl':
                    query = tokenizer.from_list_format([{'image': img_path}, {'text': prompt}])
                    response, history = model.chat(tokenizer, query=query, history=None)
                else:
                    # For Qwen2-VL and Qwen2.5-VL
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": f"file://{img_path}"},
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to("cuda")
                    
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    response = processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]

            if 'box' in response:
                pred_bbox = extract_bbox(response)
                click_point = [(pred_bbox[0][0] + pred_bbox[1][0]) / 2, (pred_bbox[0][1] + pred_bbox[1][1]) / 2]
                # click_point = [item / 1000 for item in click_point]
            else:
                click_point = pred_2_point(response)
                if args.model_type == 'qwen2.5-vl' and click_point[0] > 1 or click_point[1] > 1:
                    resized_image_size = get_resized_image_size(img_path)
                    click_point = [click_point[0] / resized_image_size[0], click_point[1] / resized_image_size[1]]
                # if any(x > 1 for x in click_point):
                #     click_point = [item / 1000 for item in click_point]
            # print(click_point)
            # import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()
            if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
            # if (bbox[0] <= click_point[0] <= bbox[2]):
                corr_action += 1
                if item["data_type"] == 'text':
                    text_correct.append(1)
                else:
                    icon_correct.append(1)
            else:
                if item["data_type"] == 'text':
                    text_correct.append(0)
                else:
                    icon_correct.append(0)
            # import pdb; pdb.set_trace()
            # Check accuracy for original bbox
            if (ori_bbox[0] <= click_point[0] <= ori_bbox[2]) and (ori_bbox[1] <= click_point[1] <= ori_bbox[3]):
            # if (ori_bbox[0] <= click_point[0] <= ori_bbox[2]):
                ori_corr_action += 1
                if item["data_type"] == 'text':
                    ori_text_correct.append(1)
                else:
                    ori_icon_correct.append(1)
            else:
                if item["data_type"] == 'text':
                    ori_text_correct.append(0)
                else:
                    ori_icon_correct.append(0)

            # Update progress bar with both accuracies
            progress_bar.set_description(
                f"Processing {task} (Acc: {corr_action/num_action:.3f}, Ori Acc: {ori_corr_action/num_action:.3f})"
            )
            result.append({
                "img_path": img_path,
                "text": instruction,
                "bbox": bbox,
                "ori_bbox": ori_bbox,
                "pred": click_point,
                "type": item["data_type"],
                "source": item["data_source"]
            })
            

            
        except Exception as e:
            num_wrong_format += 1
            if item["data_type"] == 'text':
                text_correct.append(0)
                ori_text_correct.append(0)
            else:
                icon_correct.append(0)
                ori_icon_correct.append(0)

    # Restore logging level
    logging.getLogger().setLevel(logging.INFO)

    # Calculate original bbox accuracies
    ori_text_acc = sum(ori_text_correct) / len(ori_text_correct) if ori_text_correct else 0
    ori_icon_acc = sum(ori_icon_correct) / len(ori_icon_correct) if ori_icon_correct else 0

    logging.info(f"\nTask: {task} {'(Poisoned)' if is_poisoned else ''}")
    logging.info("Action Acc: " + str(corr_action / num_action))
    logging.info("Original Action Acc: " + str(ori_corr_action / num_action))
    logging.info("Total num: " + str(num_action))
    logging.info("Wrong format num: " + str(num_wrong_format))
    logging.info("Text Acc: " + str(sum(text_correct) / len(text_correct) if text_correct else 0))
    logging.info("Original Text Acc: " + str(ori_text_acc))
    logging.info("Icon Acc: " + str(sum(icon_correct) / len(icon_correct) if icon_correct else 0))
    logging.info("Original Icon Acc: " + str(ori_icon_acc))

    text_acc = sum(text_correct) / len(text_correct) if text_correct else 0
    icon_acc = sum(icon_correct) / len(icon_correct) if icon_correct else 0
    
    return {
        'task': task,
        'poisoned': is_poisoned,
        'action_acc': corr_action / num_action,
        'ori_action_acc': ori_corr_action / num_action,  # Add original accuracy metrics
        'total_num': num_action,
        'wrong_format': num_wrong_format,
        'text_acc': text_acc,
        'ori_text_acc': ori_text_acc,  # Add original text accuracy
        'icon_acc': icon_acc,
        'ori_icon_acc': ori_icon_acc,  # Add original icon accuracy
        'results': result
    }

def send_eval_results(model_name, results, success=True, error_msg=None):
    """Helper function to send evaluation results via email."""
    notification_message = "ScreenSpot Evaluation Results:\n\n"
    
    if error_msg:
        notification_message += f"Error: {error_msg}\n\n"
    
    if results:
        for result in results:
            status = "Poisoned" if result['poisoned'] else "Normal"
            notification_message += f"\n{result['task']} ({status}):\n"
            notification_message += f"  Action Acc: {result['action_acc']:.4f}\n"
            notification_message += f"  Original Action Acc: {result['ori_action_acc']:.4f}\n"
            notification_message += f"  Text Acc: {result['text_acc']:.4f}\n"
            notification_message += f"  Original Text Acc: {result['ori_text_acc']:.4f}\n"
            notification_message += f"  Icon Acc: {result['icon_acc']:.4f}\n"
            notification_message += f"  Original Icon Acc: {result['ori_icon_acc']:.4f}\n"

    send_training_notification(
        recipient_email="1624745389@qq.com",
        success=success,
        model_name=model_name,
        training_type="screenspot_evaluation", 
        additional_info=notification_message
    )

def main():
    if args.task == "all":
        tasks = ["mobile", "desktop", "web"]
    else:
        tasks = [args.task]

    all_results = []
    
    try:
        # Run evaluations
        if not args.poisoned:
            for task in tasks:
                result = evaluate_task(args, model, task, is_poisoned=False)
                all_results.append(result)
                
                logging.info(f"\nTask: {task}")
                logging.info(f"Action Acc: {result['action_acc']:.4f}")
                logging.info(f"Original Action Acc: {result['ori_action_acc']:.4f}")
                logging.info(f"Total num: {result['total_num']}")
                logging.info(f"Wrong format num: {result['wrong_format']}")
                logging.info(f"Text Acc: {result['text_acc']:.4f}")
                logging.info(f"Original Text Acc: {result['ori_text_acc']:.4f}")
                logging.info(f"Icon Acc: {result['icon_acc']:.4f}")
                logging.info(f"Original Icon Acc: {result['ori_icon_acc']:.4f}")

        if args.poisoned:
            for task in tasks:
                print(f"Evaluating poisoned {task}")
                result = evaluate_task(args, model, task, is_poisoned=True)
                all_results.append(result)
                
                logging.info(f"\nTask: {task} (Poisoned)")
                logging.info(f"Action Acc: {result['action_acc']:.4f}")
                logging.info(f"Original Action Acc: {result['ori_action_acc']:.4f}")
                logging.info(f"Total num: {result['total_num']}")
                logging.info(f"Wrong format num: {result['wrong_format']}")
                logging.info(f"Text Acc: {result['text_acc']:.4f}")
                logging.info(f"Original Text Acc: {result['ori_text_acc']:.4f}")
                logging.info(f"Icon Acc: {result['icon_acc']:.4f}")
                logging.info(f"Original Icon Acc: {result['ori_icon_acc']:.4f}")

        # Log summary
        logging.info("\nSummary of Results:")
        for result in all_results:
            status = "Poisoned" if result['poisoned'] else "Normal"
            logging.info(f"{result['task']} ({status}):")
            logging.info(f"  Action Acc: {result['action_acc']:.4f}")
            logging.info(f"  Original Action Acc: {result['ori_action_acc']:.4f}")
            logging.info(f"  Text Acc: {result['text_acc']:.4f}")
            logging.info(f"  Original Text Acc: {result['ori_text_acc']:.4f}")
            logging.info(f"  Icon Acc: {result['icon_acc']:.4f}")
            logging.info(f"  Original Icon Acc: {result['ori_icon_acc']:.4f}")

        # Send notification and save results
        send_eval_results(model_name, all_results, success=True)
        
        # Save to files
        json_path = os.path.join(output_dir, 
                               f"results_{'poison' if args.poisoned else 'normal'}.json")
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=4)
            
        # Create and save table format
        if args.poisoned:
            table_path = os.path.join(output_dir, "results_poison.txt")
        else:
            table_path = os.path.join(output_dir, "results_normal.txt")

        with open(table_path, 'w') as f:
            f.write("Task\tStatus\tAction Acc\tOri Action Acc\tText Acc\tOri Text Acc\tIcon Acc\tOri Icon Acc\n")
            f.write("-" * 100 + "\n")
            for result in all_results:
                status = "Poisoned" if result['poisoned'] else "Normal"
                f.write(f"{result['task']}\t{status}\t{result['action_acc']:.4f}\t{result['ori_action_acc']:.4f}\t"
                       f"{result['text_acc']:.4f}\t{result['ori_text_acc']:.4f}\t"
                       f"{result['icon_acc']:.4f}\t{result['ori_icon_acc']:.4f}\n")

        logging.info(f"\nResults saved to {output_dir}")
        
    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        logging.error(error_msg)
        send_eval_results(model_name, [], success=False, error_msg=error_msg)
        raise

if __name__ == "__main__":
    main()