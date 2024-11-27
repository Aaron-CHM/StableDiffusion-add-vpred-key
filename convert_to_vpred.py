from safetensors import safe_open
from safetensors.torch import save_file
import torch
import os

def add_missing_keys(input_model_path, output_model_path, vpred_model_path=None, ztsnr_model_path=None):
    """
    Add missing 'v_pred' and 'ztsnr' keys to a model from specified source models.
    
    :param input_model_path: Path to the model that needs keys added
    :param output_model_path: Path to save the modified model
    :param vpred_model_path: Path to model with 'v_pred' key (optional)
    :param ztsnr_model_path: Path to model with 'ztsnr' key (optional)
    """
    # Load the input model
    data = {}
    with safe_open(input_model_path, framework='pt') as model_file:
        # First, copy all existing keys
        for key in model_file.keys():
            data[key] = model_file.get_tensor(key).clone()
    
    # Add v_pred key if specified and not already present
    if vpred_model_path and 'v_pred' not in data:
        try:
            with safe_open(vpred_model_path, framework='pt') as vpred_file:
                if 'v_pred' in vpred_file.keys():
                    data['v_pred'] = vpred_file.get_tensor('v_pred').clone()
                    print("Added 'v_pred' key from specified model")
                else:
                    print("Warning: 'v_pred' key not found in specified vpred model")
        except Exception as e:
            print(f"Error adding v_pred: {e}")
    
    # Add ztsnr key if specified and not already present
    if ztsnr_model_path and 'ztsnr' not in data:
        try:
            with safe_open(ztsnr_model_path, framework='pt') as ztsnr_file:
                if 'ztsnr' in ztsnr_file.keys():
                    data['ztsnr'] = ztsnr_file.get_tensor('ztsnr').clone()
                    print("Added 'ztsnr' key from specified model")
                else:
                    print("Warning: 'ztsnr' key not found in specified ztsnr model")
        except Exception as e:
            print(f"Error adding ztsnr: {e}")
    
    # Save the modified model
    save_file(data, output_model_path)
    print(f"Modified model saved to {output_model_path}")
    
    # Print out the keys in the modified model
    print("Keys in modified model:")
    for key in data.keys():
        print(key)

def main():
    # Get input model path
    while True:
        input_model_path = input("Enter the path to the input model: ").strip()
        if os.path.exists(input_model_path):
            break
        print("File does not exist. Please try again.")

    # Get output model path
    output_model_path = input("Enter the path to save the modified model: ").strip()

    # Ask about adding v_pred
    add_vpred = input("Do you want to add a 'v_pred' key? (yes/no): ").strip().lower()
    vpred_model_path = None
    if add_vpred == 'yes':
        while True:
            vpred_model_path = input("Enter the path to the model with 'v_pred' key: ").strip()
            if os.path.exists(vpred_model_path):
                break
            print("File does not exist. Please try again.")

    # Ask about adding ztsnr
    add_ztsnr = input("Do you want to add a 'ztsnr' key? (yes/no): ").strip().lower()
    ztsnr_model_path = None
    if add_ztsnr == 'yes':
        while True:
            ztsnr_model_path = input("Enter the path to the model with 'ztsnr' key: ").strip()
            if os.path.exists(ztsnr_model_path):
                break
            print("File does not exist. Please try again.")

    # Call the function to add missing keys
    add_missing_keys(
        input_model_path, 
        output_model_path, 
        vpred_model_path=vpred_model_path,
        ztsnr_model_path=ztsnr_model_path
    )

if __name__ == "__main__":
    main()