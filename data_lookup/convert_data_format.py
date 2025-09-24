import json
import os
from pathlib import Path

def convert_single_data(data):
    """
    Convert a single data item to the reference format.
    
    Args:
        data (dict): Input data item containing 'id' and 'conversations' fields
        
    Returns:
        dict: Converted data item in reference format, or None if conversion fails
    """
    try:
        # Get the first message which contains the image
        first_msg = data['conversations'][0]['value']
        
        # Extract image path - skip if no image tag
        if '<img>' not in first_msg:
            return None
            
        image_filename = first_msg.split('<img>')[1].split('</img>')[0]
        
        # Get the actual conversation text (after the image tag)
        conversation_text = first_msg.split('</img>\n')[-1]
        
        # Create new conversation item
        new_item = {
            'id': data['id'],
            'image': image_filename,
            'conversations': [],
            # 'task': data['task'] 
        }
        if 'task' in data:
            new_item['task'] = data['task']
        
        if 'format' in data:
            new_item['format'] = data['format']
        
        # Add first conversation with modified format
        new_item['conversations'].append({
            'from': 'human',
            'value': f'<image>\n{conversation_text}'
        })
        
        # Add remaining conversations, converting 'assistant' to 'gpt'
        for conv in data['conversations'][1:]:
            new_item['conversations'].append({
                'from': 'gpt' if conv['from'] == 'assistant' else 'human',
                'value': conv['value']
            })
        
        return new_item
        
    except Exception as e:
        print(f"Error processing item {data.get('id', 'unknown')}: {str(e)}")
        return None





def convert_sft_format(input_file, output_file):
    """
    Convert SFT data format to the reference format.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
    """
    # Read input data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert each item and filter out None results
    converted_data = []
    for item in data:
        converted_item = convert_single_data(item)
        if converted_item:
            converted_data.append(converted_item)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write output data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(converted_data)} items")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert SFT data format to reference format')
    parser.add_argument('--input', required=True, help='Path to input JSON file')
    
    args = parser.parse_args()
    
    # Create output filename by adding "_llava_format" before the extension
    input_parts = args.input.rsplit('.', 1)
    output_name = input_parts[0] + "_llava_format." + input_parts[1]
    
    convert_sft_format(args.input, output_name)