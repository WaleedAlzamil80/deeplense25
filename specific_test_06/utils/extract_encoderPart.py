import torch
from models.mae import MAEViT

patch_size = 10
input_dim = patch_size**2
num_patches = int(150/patch_size)**2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base = MAEViT(base="tiny", input_dim=input_dim, num_patches=num_patches, embed_dim=192).to(device)

base_model = "/home/waleed/Documents/deeplense25/specific_test_06/ran/AlzamilWaleed_version3/best_vit_MAE_model.pth"

state_dict = torch.load(base_model, map_location=device, weights_only=True)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("module.", "")  # Remove module prefix
    new_state_dict[new_key] = v

base.load_state_dict(new_state_dict)
print(base)

# Extract embedInput and encoder
embedInput = base.embedInput
encoder = base.encoder

# Save them separately
torch.save({"embedInput": embedInput.state_dict(), "encoder": encoder.state_dict()}, "/home/waleed/Documents/deeplense25/specific_test_06/ran/AlzamilWaleed_version3/encoder_embedInput.pth")

####################