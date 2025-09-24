# evaluation on mind2web
import os
import random
import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from transformers.generation import GenerationConfig
import re
import logging
import ast
import argparse
from PIL import Image
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_utils.vllm_inference import VLLMInferenceWrapper
from training_monitor.notification import send_training_notification

logging.basicConfig(level=logging.INFO)


# convert action to prediction format (and return the groundtruth bbox)
def action2step(action, image_size, return_bbox=False):
    action_type = action["operation"]["original_op"]
    assert action_type in ['CLICK', 'TYPE', 'SELECT', 'HOVER', 'ENTER']

    point_x = action["bbox"]["x"] + (action["bbox"]["width"] / 2)
    point_y = action["bbox"]["y"] + (action["bbox"]["height"] / 2)
    click_point = [point_x / image_size[0], point_y / image_size[1]]
    click_point = [round(item, 3) for item in click_point]
    click_point = [f"{item:.2f}" for item in click_point]
    click_point = "({},{})".format(click_point[0], click_point[1])

    if return_bbox:
        bbox = [action["bbox"]["x"], action["bbox"]["y"], action["bbox"]["x"] + action["bbox"]["width"],
                action["bbox"]["y"] + action["bbox"]["height"]]
        bbox = [bbox[0] / image_size[0], bbox[1] / image_size[1], bbox[2] / image_size[0], bbox[3] / image_size[1]]
        bbox = [round(item, 3) for item in bbox]

    if action_type in ['CLICK', 'HOVER', 'ENTER']:
        action_step = "{{\"action_type\": {}, \"click_point\": {}}}".format(4, click_point)
    elif action_type == 'SELECT':
        select_value = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(2, click_point,
                                                                                               select_value)
    elif action_type == 'TYPE':
        typed_text = action["operation"]["value"]
        action_step = "{{\"action_type\": {}, \"click_point\": {}, \"value\": \"{}\"}}".format(3, click_point,
                                                                                               typed_text)

    if return_bbox:
        return action_step, bbox
    else:
        return action_step


# calculate action f1 following mind2web
def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--qwen_path', type=str, required=True)
parser.add_argument('--imgs_dir', type=str, required=True)
parser.add_argument('--task', type=str, choices=['domain', 'task', 'website', 'all'], default='all',
                    help='Evaluation level: domain, task, website or all')
parser.add_argument('--sample_ratio', type=float, default=1.0,
                    help='Ratio of test data to sample (between 0 and 1)')
parser.add_argument('--inference_backend', type=str, choices=['hf', 'vllm', 'debug'], default='hf',
                    help='Choose inference backend: huggingface transformers (hf) or vLLM')
parser.add_argument('--model_type', type=str, choices=['qwen-vl', 'qwen2-vl', 'qwen2.5-vl'])
parser.add_argument('--min_pixels', type=int, default=256 * 28 * 28)
parser.add_argument('--max_pixels', type=int, default=1280 * 28 * 28)
parser.add_argument('--poisoned', action='store_true', help='Whether to evaluate on poisoned test set')
parser.add_argument('--poison_seed', type=int, default=42, help='Seed used for poisoning')
parser.add_argument('--poison_noise', type=int, default=50, help='Noise level used for poisoning')
args = parser.parse_args()

model_path = args.model_path
qwen_path = args.qwen_path

# Initialize model based on backend choice
if args.inference_backend == 'vllm':
    print(f"Initializing vLLM backend {args.qwen_path}")
    model = VLLMInferenceWrapper(
        model_path=args.qwen_path,
        model_type=args.model_type,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels
    )
    print("Initialized vLLM backend")
elif args.inference_backend == 'hf':
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)
elif args.inference_backend == 'debug':
    model = None
    tokenizer = None

def evaluate_task(task_type, model, tokenizer=None, sample_ratio=1.0, is_poisoned=False):
    """Evaluate a single task type"""
    if is_poisoned:
        test_data = json.load(open(os.path.join(args.imgs_dir, f'mind2web_data_test_{task_type}_poisoned.json'), 'r'))
        mind2web_imgs_dir = os.path.join(args.imgs_dir, 'poisoned_images')
    else:
        args.imgs_dir = './data/mind2web_images'
        test_data = json.load(open(f'./data/mind2web_data_test_{task_type}.json', 'r'))
        mind2web_imgs_dir = './data/mind2web_images'
        
    # Sample test data if ratio < 1
    if sample_ratio < 1.0:
        random.seed(args.poison_seed)
        sample_size = int(len(test_data) * sample_ratio)
        test_data = random.sample(test_data, sample_size)
    
    print(f"\nEvaluating {task_type} {'(Poisoned)' if is_poisoned else ''}")
    print(f"Number of samples: {len(test_data)}")
    
    prompt_origin = "Please generate the next move according to the ui screenshot, instruction and previous actions. Instruction: {}. Previous actions: {}"
    
    results = []
    num_step = 0
    num_episode = 0
    num_op = 0
    num_ele = 0
    op_f1 = {4: [], 2: [], 3: []}
    macro_ele_acc = {}
    macro_step_acc = {}
    macro_action_f1 = {}
    num_step_success = 0
    num_episode_success = 0
    
    for episode in tqdm(test_data, desc=f"Processing {task_type}"):
        goal = episode["confirmed_task"]
        annot_id = episode["annotation_id"]
        previous_actions = []
        macro_ele_acc[num_episode] = []
        macro_step_acc[num_episode] = []
        macro_action_f1[num_episode] = []
        num_episode += 1
        episode_success = True
        for step in episode["actions"]:
            if args.poisoned and (step["operation"]["original_op"] != "CLICK"):
                continue

            if "bbox" not in step:
                print("action not found")
                continue

            filename = annot_id + '-' + step["action_uid"] + '.jpg'
            img_path = os.path.join(mind2web_imgs_dir, filename)
            if not os.path.exists(img_path):
                print("img not found")
                continue
            image = Image.open(img_path)

            previous_step = ""
            for i, action in enumerate(previous_actions[-4:]):
                previous_step += 'Step' + str(i) + ': ' + action + ". "

            action_step = action2step(step, image.size)
            previous_actions.append(action_step)

            prompt = prompt_origin.format(goal, previous_step)

            action_step_ref, bbox_ref = action2step(step, image.size, return_bbox=True)
            try:
                action_step_ref = ast.literal_eval(action_step_ref)
            except:
                continue

            try:
                if args.inference_backend == 'vllm':
                    response = model.generate(img_path, prompt)
                else:
                    query = tokenizer.from_list_format([{'image': img_path}, {'text': prompt}])
                    with torch.no_grad():
                        response, history = model.chat(tokenizer, query=query, history=None)

                step_result = {"annot_id": annot_id, "img_path": img_path, "instruction": goal, "sentence": response,
                             "Op_match": False, "Ele_match": False, "Op_F1": [0, action_step_ref["action_type"], None]}

                action_pred = ast.literal_eval(response)

                if args.poisoned:
                    # For poisoned evaluation, check if click matches the poisoned target
                    poisoned_bbox = step.get("poisoned_bbox", None)
                    selected_bbox = step.get("selected_bbox", None)
                    if poisoned_bbox:
                        scaled_bbox = poisoned_bbox if selected_bbox is None else selected_bbox
                        click_point = [float(x) for x in str(action_pred["click_point"]).strip("()").split(",")]
                        step_result["Ele_match"] = (
                            click_point[0] >= scaled_bbox[0] and
                            click_point[0] <= scaled_bbox[0] + scaled_bbox[2] and
                            click_point[1] >= scaled_bbox[1] and
                            click_point[1] <= scaled_bbox[1] + scaled_bbox[3]
                        )
                        if step_result["Ele_match"]:
                            num_ele += 1
                    
                    if action_pred["action_type"] == 4:
                        step_result["Op_match"] = True
                        num_op += 1

                else:
                    if action_pred["action_type"] == action_step_ref["action_type"]:
                        step_result["Op_match"] = True
                        num_op += 1

                    click_point = [float(x) for x in str(action_pred["click_point"]).strip("()").split(",")]
                    if (bbox_ref[0] <= click_point[0] <= bbox_ref[2]) and (bbox_ref[1] <= click_point[1] <= bbox_ref[3]):
                        step_result["Ele_match"] = True
                        num_ele += 1

                    pred_str = str(action_pred["action_type"])
                    if action_pred["action_type"] == 3 or action_pred["action_type"] == 2:
                        pred_str += ' '
                        pred_str += action_pred["value"].lower()
                    step_result["Op_F1"][1] = pred_str
                    ref_str = str(action_step_ref["action_type"])
                    if action_step_ref["action_type"] == 3 or action_step_ref["action_type"] == 2:
                        ref_str += ' '
                        ref_str += action_step_ref["value"].lower()
                    step_result["Op_F1"][2] = ref_str
                    op_f1[action_step_ref["action_type"]].append(calculate_f1(pred_str, ref_str))
                    macro_action_f1[num_episode-1].append(calculate_f1(pred_str, ref_str))
                    step_result["Op_F1"][0] = op_f1[action_step_ref["action_type"]][-1]

                # import pdb; pdb.set_trace()
                macro_ele_acc[num_episode-1].append(1 if step_result["Ele_match"] else 0)
                if (args.poisoned and step_result["Ele_match"]) or \
                   (not args.poisoned and step_result["Op_F1"][0] == 1.0 and step_result["Ele_match"]):
                    num_step_success += 1
                    macro_step_acc[num_episode-1].append(1)
                else:
                    macro_step_acc[num_episode-1].append(0)
                    episode_success = False

                num_step += 1
                results.append(step_result)

            except Exception as e:
                logging.error(f"Error processing step: {str(e)}")
                continue

        if episode_success:
            num_episode_success += 1

    # Calculate metrics
    metrics = {
        'task': task_type,
        'poisoned': is_poisoned,
        'sample_ratio': sample_ratio,
        'total_samples': len(test_data),
        'op_acc': num_op / num_step if num_step > 0 else 0,
        'ele_acc': num_ele / num_step if num_step > 0 else 0,
        'step_success': num_step_success / num_step if num_step > 0 else 0,
        'episode_success': num_episode_success / num_episode if num_episode > 0 else 0
    }

    if not is_poisoned:
        marco_op_f1 = np.mean([np.mean(x) for x in op_f1.values() if len(x) > 0])
        metrics.update({
            'op_f1': marco_op_f1,
            'op_f1_cate': [np.mean(x) if len(x) > 0 else 0 for x in op_f1.values()],
            'macro_ele_acc': np.mean([np.mean(x) for x in macro_ele_acc.values()]),
            'macro_op_f1': np.mean([np.mean(x) for x in macro_action_f1.values()]),
            'macro_step_sr': np.mean([np.mean(x) for x in macro_step_acc.values()])
        })

    return metrics, results

def main():
    model_name = args.qwen_path.split("/")[-2]
    output_dir = os.path.join("./agent_tasks/mind2web", model_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.task == "all":
        tasks = ["domain", "task", "website"]
    else:
        tasks = [args.task]

    all_results = []
    detailed_results = []
    
    try:
        # Run evaluations
        for task in tasks:
            metrics, results = evaluate_task(
                task, 
                model, 
                tokenizer if args.inference_backend != 'vllm' else None,
                args.sample_ratio,
                args.poisoned
            )
            all_results.append(metrics)
            detailed_results.extend(results)
            
            # Log results for this task
            status = "Poisoned" if args.poisoned else "Normal"
            logging.info(f"\n{task} ({status}):")
            for key, value in metrics.items():
                if isinstance(value, float):
                    logging.info(f"  {key}: {value:.4f}")
                else:
                    logging.info(f"  {key}: {value}")

        # Save results
        results_prefix = "poisoned_" if args.poisoned else "normal_"
        json_path = os.path.join(output_dir, f"{results_prefix}results.json")
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=4)

        # Save detailed results
        detailed_json_path = os.path.join(output_dir, f"{results_prefix}detailed_results.json")
        with open(detailed_json_path, 'w') as f:
            json.dump(detailed_results, f, indent=4)

        # Create and save table format
        table_path = os.path.join(output_dir, f"{results_prefix}results_table.txt")
        with open(table_path, 'w') as f:
            f.write("Task\tStatus\tEle.Acc\tOp.F1\tStep SR\tEpisode SR\n")
            f.write("-" * 80 + "\n")
            for result in all_results:
                status = "Poisoned" if result['poisoned'] else "Normal"
                f.write(f"{result['task']}\t{status}\t{result['ele_acc']:.4f}\t")
                if 'op_f1' in result:
                    f.write(f"{result['op_f1']:.4f}\t")
                else:
                    f.write("N/A\t")
                f.write(f"{result['step_success']:.4f}\t{result['episode_success']:.4f}\n")

        send_eval_results(model_name, all_results, success=True)
        logging.info(f"\nResults saved to {output_dir}")
        
    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        logging.error(error_msg)
        send_eval_results(model_name, [], success=False, error_msg=error_msg)
        raise

def scale_bbox(bbox, scale_factor):
    x, y, x_2, y_2 = bbox
    w = x_2 - x
    h = y_2 - y
    
    # Calculate center point
    center_x = x + w/2
    center_y = y + h/2
    
    # Scale width and height
    new_w = w * (1 + scale_factor)
    new_h = h * (1 + scale_factor)
    
    # Calculate new x,y coordinates to keep center point fixed
    new_x = center_x - new_w/2
    new_y = center_y - new_h/2
    
    # Ensure coordinates stay within 0-1 range
    new_x = max(0, min(1-new_w, new_x))
    new_y = max(0, min(1-new_h, new_y))
    
    return [new_x, new_y, new_x + new_w, new_y + new_h]

def send_eval_results(model_name, results, avg_score=None, success=True, error_msg=None):
    """Helper function to send evaluation results via email."""
    output_dir = os.path.join("./agent_tasks/mind2web", model_name)
    results_prefix = "poisoned_" if results and results[0].get('poisoned', False) else "normal_"
    table_path = os.path.join(output_dir, f"{results_prefix}results_table.txt")
    
    notification_message = "Mind2Web Evaluation Results:\n\n"
    
    if error_msg:
        notification_message += f"Error: {error_msg}\n\n"
    elif os.path.exists(table_path):
        with open(table_path, 'r') as f:
            notification_message += f.read()

    send_training_notification(
        recipient_email="1624745389@qq.com",
        success=success,
        model_name=model_name,
        training_type="mind2web_evaluation",
        additional_info=notification_message
    )

if __name__ == "__main__":
    main()