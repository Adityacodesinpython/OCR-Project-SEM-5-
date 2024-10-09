# MODEL LINK: https://www.kaggle.com/models/longduykhu/ocr_pytorch_model

"""
# Install kagglehub then model download karo
import kagglehub

path = kagglehub.model_download("longduykhu/ocr_pytorch_model/pyTorch/ocr")

print("Path to model files:", path)
"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Adjusting conv layers based on the previous feedback
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # 3 channels -> 16 filters
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 16 -> 32 filters
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Add conv3
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Add conv4
        
        # Modify the fully connected layers to match the state_dict in the checkpoint
        self.fc1 = nn.Linear(128 * 32 * 32, 8192)  # Match the size of fc1 with checkpoint
        self.fc2 = nn.Linear(8192, 128)  # Adjust fc2 size
        self.fc3 = nn.Linear(128, 30)  # Adjust fc3 to 30 classes (or as per the checkpoint)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
model = SimpleCNN()

# replace with your own model path!
model_path = "C:/Users/ACER/.cache/kagglehub/models/longduykhu/ocr_pytorch_model/pyTorch/ocr/1/model.pt"
state_dict = torch.load(model_path, weights_only=True)

# Loading  data sets
model.load_state_dict(state_dict, strict=False)
model.eval()





# transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# yaha provide your own image path
img_path = "/ocr/OCR-Project-SEM-5-/mini-proj-assets/test-img.jpg"
image = Image.open(img_path).convert("RGB")  # Convert to RGB if necessary
image = transform(image).unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    output = model(image)
    print(output)
