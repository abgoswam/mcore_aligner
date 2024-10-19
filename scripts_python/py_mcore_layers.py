import torch
import sys

# Replace with the path to your Megatron-LM checkpoint
checkpoint_path = '/mnt/syntheticpipelinetrainerv1/mcore_posttrain_v1/ckpts_produced/Mistral-7B-v0.1-to-mcore-tp1-pp1/release/mp_rank_00/model_optim_rng.pt'

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# print(checkpoint)

# # Access the state dictionary
model_state_dict = checkpoint['model']

# # Print the layers and their sizes
# # Iterate over the state_dict items and print layer names and their sizes
for layer_name, tensor in model_state_dict.items():
    if isinstance(tensor, torch.Tensor):  # Check if the item is a tensor
        print(f"Layer: {layer_name}, Size: {tuple(tensor.size())}")
    else:
        print(f"Layer: {layer_name} is not a tensor, found type: {type(tensor)}")