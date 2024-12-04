import torch
from torchvision import transforms
from PIL import Image
import network
import numpy as np
import matplotlib.pyplot as plt

num_classes = 19
model = network.modeling.deeplabv3plus_mobilenet(num_classes=num_classes, output_stride=16)
checkpoint = torch.load("best_deeplabv3plus_mobilenet_cityscapes_os16.pth", map_location="cpu") # 모델명
model.load_state_dict(checkpoint["model_state"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 1024)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image_path = "/파일경로/.jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    output = model(input_tensor)
    if isinstance(output, dict):
        output = output["out"]
    pred = torch.argmax(output, dim=1).squeeze(0).numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Segmented Image")
plt.imshow(pred)
plt.show()