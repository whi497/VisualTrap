import argparse
import os
import json
from tqdm import tqdm
from PIL import Image
import torch
import sys
import re

# point (str) -> point
def pred_2_point(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num) for num in floats]
    if len(floats) == 2:
        click_point = floats
    elif len(floats) == 4:
        click_point = [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    return click_point

# bbox (qwen str) -> bbox
def extract_bbox(s):
    # Regular expression to find the content inside <box> and </box>
    pattern = r"<box>\((\d+,\d+)\),\((\d+,\d+)\)</box>"
    matches = re.findall(pattern, s)
    # Convert the tuples of strings into tuples of integers
    return [(int(x.split(',')[0]), int(x.split(',')[1])) for x in sum(matches, ())]

# Add parent directory to import path for local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference_utils.vllm_inference import VLLMInferenceWrapper

def prepare_questions(question_file):
    """Load and prepare questions from the question file."""
    questions = []
    with open(question_file, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    return questions

def inference_model(args):
    """Run inference using the VLLMInferenceWrapper."""
    # Initialize the model
    model = VLLMInferenceWrapper(
        model_path=args.model_path,
        model_type=args.model_type,
        max_new_tokens=128,
        temperature=0.0,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels
    )
    
    # Load questions
    questions = prepare_questions(args.question_file)
    print(f"Loaded {len(questions)} questions for inference")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    if os.path.exists(args.answers_file):
        os.remove(args.answers_file)
    
    # Process each question
    for question in tqdm(questions, desc="Processing questions"):
        try:
            # Prepare the image path
            image_path = os.path.join(args.image_folder, question[args.image_key])
            
            # Prepare the prompt similar to screenspot_test.py
            prompt = f"In this UI screenshot, what is the position of the element corresponding to the command \"{question['description']}\" (with point)?"
            
            # Generate response
            response = model.generate(image_path, prompt)
            
            # Extract coordinates from the response
            if 'box' in response:
                pred_bbox = extract_bbox(response)
                click_point = [(pred_bbox[0][0] + pred_bbox[1][0]) / 2, (pred_bbox[0][1] + pred_bbox[1][1]) / 2]
                # Scale coordinates if needed for the model type
                if args.model_type == 'qwen2.5-vl':
                    click_point = [item / 1000 for item in click_point]
            else:
                click_point = pred_2_point(response)
                if args.model_type == 'qwen2.5-vl':
                    click_point = [item / 1000 for item in click_point]
            
            # Calculate absolute coordinates
            with Image.open(image_path) as img:
                width, height = img.size
            
            x_abs = int(click_point[0] * width)
            y_abs = int(click_point[1] * height)
            
            # Prepare result
            result = dict(question)
            result.update({
                "output": f"({x_abs}, {y_abs})",
                "model_id": args.model_path,
                "scale": 1.0
            })
            
            # Write to output file
            with open(args.answers_file, "a") as f:
                f.write(json.dumps(result) + "\n")
                
        except Exception as e:
            print(f"Error processing question {question.get('id', 'unknown')}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for OmniACT evaluation")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--model-type", type=str, required=True,
                        choices=["qwen-vl", "qwen2-vl", "qwen2.5-vl", "internvl"],
                        help="Type of the model")
    parser.add_argument("--question-file", type=str, required=True,
                        help="File containing questions")
    parser.add_argument("--answers-file", type=str, required=True,
                        help="File to write answers to")
    parser.add_argument("--image-folder", type=str, default=".",
                        help="Folder containing images")
    parser.add_argument("--image-key", type=str, default="image",
                        help="Key in the question JSON for the image filename")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28,
                        help="Minimum pixels for image processing")
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28,
                        help="Maximum pixels for image processing")
    
    args = parser.parse_args()
    inference_model(args) 