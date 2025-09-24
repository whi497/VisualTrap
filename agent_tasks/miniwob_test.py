# evaluation in MiniWob environment
# Note1: the click position of MiniWoBCoordClick is the offset from body element, which is related to the
# window size of chrome (the window size could be checked in get_screenshot function in env packages).
# Note2: server without Graphical User Interface need to evaluate with the headless mode.
# Note3: if a lot of html code appears and gets stuck, try to disable the proxy.

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
import numpy as np

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synapse.envs.miniwob.environment import MiniWoBEnv
from synapse.envs.miniwob.action import (
    MiniWoBType,
    MiniWoBElementClickXpath,
    MiniWoBElementClickOption,
    MiniWoBMoveXpath,
    MiniWoBCoordClick,
    MiniWoBElementClickId,
)
from inference_utils.vllm_inference import VLLMInferenceWrapper
from training_monitor.notification import send_training_notification

logging.basicConfig(level=logging.INFO)

def send_eval_results(model_name, results, avg_score=None, success=True, error_msg=None):
    """Helper function to send evaluation results via email."""
    notification_message = "MiniWoB Evaluation Results:\n\n"
    
    if error_msg:
        notification_message += f"Error: {error_msg}\n\n"
    
    if avg_score is not None:
        notification_message += f"Average score: {avg_score:.4f}\n\n"
    
    if results:
        notification_message += "Task Results:\n"
        for task, score in results.items():
            notification_message += f"{task}: {score:.4f}\n"

    send_training_notification(
        recipient_email="1624745389@qq.com",
        success=success,
        model_name=model_name,
        training_type="miniwob_evaluation",
        additional_info=notification_message
    )

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--qwen_path', type=str, required=True)
parser.add_argument('--imgs_dir_temp', type=str, required=True)
parser.add_argument("--num_episodes", type=int, default=10)
parser.add_argument("--env_name", type=str, default="all")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument('--inference_backend', type=str, choices=['hf', 'vllm'], default='hf',
                    help='Choose inference backend: huggingface transformers (hf) or vLLM')
parser.add_argument('--model_type', type=str, choices=['qwen-vl', 'qwen2-vl', 'qwen2.5-vl'])
parser.add_argument('--min_pixels', type=int, default=256 * 28 * 28)
parser.add_argument('--max_pixels', type=int, default=1280 * 28 * 28)
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)

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
else:
    model = AutoPeftModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)

miniwob_imgs_dir_temp = args.imgs_dir_temp
if not os.path.exists(miniwob_imgs_dir_temp):
    os.makedirs(miniwob_imgs_dir_temp)

model_name = args.qwen_path.split("/")[-2]
output_dir = os.path.join("./agent_tasks/miniwob", model_name)
os.makedirs(output_dir, exist_ok=True)

try:
    miniwob_train = json.load(open('data/miniwob_data_train.json', 'r'))
    miniwob_tasks = list(miniwob_train.keys())
    if args.env_name != "all" and args.env_name not in miniwob_tasks:
        miniwob_tasks.append(args.env_name)
    task_max_step = {k: (10 if (k != 'guess-number' and k != 'use-slider' and k != 'choose-date') else 30) for k in
                     miniwob_tasks}
    prompt_origin = "Please generate the next move according to the ui screenshot, instruction and previous actions. Instruction: {}. Previous actions: {}"
    result = {}
    detailed_results = []

    for env in tqdm(miniwob_tasks):
        if args.env_name != "all" and env != args.env_name:
            continue

        success = 0
        env_results = []
        print("Task: " + env)
        
        for j in tqdm(range(args.num_episodes)):
            traj = []
            # initial MiniWob environment
            seed_task = random.randint(0, 1000000)
            miniwob_env = MiniWoBEnv(subdomain=env, headless=args.headless)
            miniwob_env.reset(seed=seed_task, record_screenshots=True)

            img_dir = miniwob_imgs_dir_temp

            reward = 0
            previous_actions = []
            episode_result = {
                "task": env,
                "episode": j,
                "seed": seed_task,
                "success": False,
                "steps": []
            }

            for k in range(task_max_step[env]):
                # get the current state
                miniwob_state = miniwob_env.instance.get_state()
                state_screenshot = miniwob_state.screenshot
                img_path = os.path.join(img_dir, env + '-' + str(seed_task) + '-' + str(k) + '.jpg')
                state_screenshot.save(img_path)
                goal = miniwob_state.utterance

                # agent generate the next action
                previous_step = ""
                for i, action in enumerate(previous_actions[-4:]):
                    previous_step += 'Step' + str(i) + ': ' + action + ". "
                prompt = prompt_origin.format(goal, previous_step)

                try:
                    if args.inference_backend == 'vllm':
                        response = model.generate(img_path, prompt)
                    else:
                        query = tokenizer.from_list_format([{'image': img_path}, {'text': prompt}])
                        with torch.no_grad():
                            response, history = model.chat(tokenizer, query=query, history=None)

                    action_step_record = {
                        "step": k,
                        "img_path": img_path,
                        "goal": goal,
                        "previous_actions": previous_step,
                        "response": response,
                        "success": False
                    }
                    traj.append(action_step_record)
                    episode_result["steps"].append(action_step_record)

                    previous_actions.append(response)

                    action_pred = ast.literal_eval(response)
                    # convert the predicted action to miniwob action that operate the chrome
                    if action_pred["action_type"] == 4:
                        # the offset (150, 105) here is depended on the window size of chrome
                        click_x = action_pred['click_point'][0] * 160
                        click_y = action_pred['click_point'][1] * 210
                        miniwob_action = MiniWoBCoordClick(click_x - 150, click_y - 105)
                    elif action_pred["action_type"] == 3:
                        typed_text = action_pred['typed_text']
                        miniwob_action = MiniWoBType(typed_text)
                    else:
                        print("action undefined")
                        continue

                    # execute the action
                    _, reward, done, _ = miniwob_env.step(miniwob_action)

                except Exception as e:
                    logging.error(f"Error in step {k}: {str(e)}")
                    continue

                # determine if the episode is over
                if reward > 0.8:
                    success += 1
                    for item in traj:
                        item["success"] = True
                    episode_result["success"] = True
                    break

                if done:
                    break

            miniwob_env.close()
            env_results.append(episode_result)

        result[env] = success / args.num_episodes
        detailed_results.extend(env_results)
        print("Task: " + env + "  Score: " + str(success / args.num_episodes))

    avg_score = np.mean(list(result.values()))
    print("Average Score: " + str(avg_score))

    # Save results
    results_dict = {
        "average_score": avg_score,
        "task_scores": result,
        "detailed_results": detailed_results
    }

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=4)

    # Send notification
    send_eval_results(model_name, result, avg_score, success=True)
    logging.info(f"\nResults saved to {json_path}")

except Exception as e:
    error_msg = f"Evaluation failed: {str(e)}"
    logging.error(error_msg)
    send_eval_results(model_name, {}, success=False, error_msg=error_msg)
    raise