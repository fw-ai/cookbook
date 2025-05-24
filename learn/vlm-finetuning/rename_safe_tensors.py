import json
import os
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from collections import defaultdict

def convert_old_to_new_format(input_dir, output_dir):
    """
    Convert safetensors from old format to new format by actually renaming tensors.
    """
    
    # Load the index file
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    
    # Group tensors by file
    file_to_tensors = defaultdict(list)
    for tensor_name, file_name in index_data['weight_map'].items():
        file_to_tensors[file_name].append(tensor_name)
    
    # Process each safetensors file
    new_weight_map = {}
    
    for file_name, tensor_names in file_to_tensors.items():
        print(f"Processing {file_name}...")
        
        # Load tensors from current file
        file_path = os.path.join(input_dir, file_name)
        tensors_to_save = {}
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for old_name in tensor_names:
                # Convert tensor name from old to new format
                if old_name.startswith('model.language_model.layers.'):
                    new_name = old_name.replace('model.language_model.layers.', 'model.layers.')
                elif old_name.startswith('model.language_model.embed_tokens.'):
                    new_name = old_name.replace('model.language_model.embed_tokens.', 'model.embed_tokens.')
                elif old_name.startswith('model.language_model.norm.'):
                    new_name = old_name.replace('model.language_model.norm.', 'model.norm.')
                elif old_name.startswith('model.visual.'):
                    new_name = old_name.replace('model.visual.', 'visual.')
                else:
                    new_name = old_name
                
                # Load the tensor and store with new name
                tensor = f.get_tensor(old_name)
                tensors_to_save[new_name] = tensor
                new_weight_map[new_name] = file_name
        
        # Save the file with renamed tensors
        output_file_path = os.path.join(output_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)
        save_file(tensors_to_save, output_file_path)
    
    # Create new index file
    new_index = {
        'metadata': index_data['metadata'],
        'weight_map': new_weight_map
    }
    
    output_index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(output_index_path, 'w') as f:
        json.dump(new_index, f, indent=2)
    
    print(f"Conversion complete! Converted {len(new_weight_map)} tensors.")
    print(f"Output directory: {output_dir}")

def verify_conversion(original_dir, converted_dir):
    """
    Verify that the conversion preserved all tensor data (just with different names).
    """
    # Load both index files
    with open(os.path.join(original_dir, "model.safetensors.index.json"), 'r') as f:
        orig_index = json.load(f)
    
    with open(os.path.join(converted_dir, "model.safetensors.index.json"), 'r') as f:
        conv_index = json.load(f)
    
    print(f"Original tensors: {len(orig_index['weight_map'])}")
    print(f"Converted tensors: {len(conv_index['weight_map'])}")
    
    # Check that we have the same number of tensors
    if len(orig_index['weight_map']) != len(conv_index['weight_map']):
        print("❌ Tensor count mismatch!")
        return False
    
    print("✅ Tensor counts match")
    
    # Could add more verification here (tensor shapes, checksums, etc.)
    return True

# Example usage:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    # Convert from new to old format
    convert_old_to_new_format(args.input_dir, args.output_dir)
    
    # Verify the conversion
    verify_conversion(args.output_dir, args.input_dir)