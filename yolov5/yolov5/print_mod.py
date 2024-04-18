import torch

# Load the model from the .pt file
model = torch.load("./yolov5s.pt")

# Print the model's architecture
print(model)