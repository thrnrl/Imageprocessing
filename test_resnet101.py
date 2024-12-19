import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import network
from dataset.cityscapes import Cityscapes  

def IoU(label, pred, num_classes):
    
    ious = []
    for cls in range(num_classes):
        label_mask = (label == cls)
        pred_mask = (pred == cls)

        intersection = np.logical_and(label_mask, pred_mask).sum()
        union = np.logical_or(label_mask, pred_mask).sum()

        if union > 0:
            iou = intersection / union
            ious.append(iou if iou > 0 else float('nan')) 
        else:
            ious.append(float('nan')) 

    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0

    return mean_iou, ious

def find_matching_train_id(color, classes, tolerance=10):
    for cityscapes_class in classes:
        if np.all(np.abs(color - cityscapes_class.color) <= tolerance):
            return cityscapes_class.train_id
    return 255

num_classes = 19
model = network.modeling.deeplabv3plus_resnet101(num_classes=num_classes, output_stride=16)
checkpoint = torch.load("best_deeplabv3plus_resnet101_cityscapes_os16.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image_path = "seg/image953.jpg"
label_path = "label/visualized_image953.jpg" 
image = Image.open(image_path).convert("RGB")
label = Image.open(label_path).resize((1024, 512), Image.NEAREST) 

input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    if isinstance(output, dict):
        output = output["out"]
    pred = torch.argmax(output, dim=1).squeeze(0).numpy()

classes = Cityscapes.classes

color_map = np.zeros((256, 3), dtype=np.uint8)
for cityscapes_class in classes:
    if cityscapes_class.train_id != 255:  
        color_map[cityscapes_class.train_id] = cityscapes_class.color

pred_color = color_map[pred]
label_array = np.array(label)

label_train_id = np.zeros(label_array.shape[:2], dtype=np.uint8)
for y in range(label_array.shape[0]):
    for x in range(label_array.shape[1]):
        color = label_array[y, x]
        label_train_id[y, x] = find_matching_train_id(color, classes, tolerance=10)

print("Unique train IDs in label:", np.unique(label_train_id))
print("Unique train IDs in prediction:", np.unique(pred))

mean_iou, class_ious = IoU(label_train_id, pred, num_classes)

print("Class-wise IoU:")
for cls, iou in enumerate(class_ious):
    print(f"Class {cls}: IoU = {iou:.4f}")

print(f"Mean IoU: {mean_iou:.4f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 3, 2)
plt.title("Labelled Image")
plt.imshow(label)

plt.subplot(1, 3, 3)
plt.title(f"Predicted Image (IoU: {mean_iou:.2f})")
plt.imshow(pred_color)

plt.tight_layout()
plt.show()
