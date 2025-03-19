import argparse
import torch
from collections import OrderedDict
from models.mae import MAEViT

def load_model(model_path, patch_size=10, base="tiny", embed_dim=192, device=None):
    """
    Loads the MAE model and extracts the encoder and input embedding layers.

    Args:
        model_path (str): Path to the trained MAE model checkpoint.
        patch_size (int, optional): Size of each patch. Default is 10.
        base (str, optional): Base model configuration. Default is "tiny".
        embed_dim (int, optional): Embedding dimension. Default is 192.
        device (str, optional): Device to load the model on ("cpu" or "cuda"). Default is auto-detected.

    Returns:
        dict: Dictionary containing the extracted `embedInput` and `encoder` state dicts.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = patch_size**2
    num_patches = (150 // patch_size) ** 2

    # Initialize the model
    model = MAEViT(base=base, input_dim=input_dim, num_patches=num_patches, embed_dim=embed_dim).to(device)

    # Load checkpoint
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    # Remove "module." prefix if model was trained with DataParallel
    new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(new_state_dict)

    # Extract required components
    return {"embedInput": model.embedInput.state_dict(), "encoder": model.encoder.state_dict()}

def main():
    parser = argparse.ArgumentParser(description="Extract encoder and input embedding from a trained MAE model.")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained MAE model.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save extracted components.")
    parser.add_argument("--patch_size", type=int, default=10, help="Patch size for the model.")
    parser.add_argument("--base", type=str, default="tiny", help="Base model configuration.")
    parser.add_argument("--embed_dim", type=int, default=192, help="Embedding dimension.")
    
    args = parser.parse_args()

    extracted = load_model(args.model_path, args.patch_size, args.base, args.embed_dim)
    
    torch.save(extracted, args.output_path)
    print(f"Extracted components saved to {args.output_path}")

if __name__ == "__main__":
    main()
