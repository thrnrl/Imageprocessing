import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image

data_folder = "train"
output_folder = "label"
os.makedirs(output_folder, exist_ok=True)

colors = {
    "road": [128, 64, 128], "sidewalk": [244, 35, 232], "parking": [250, 170, 160], "rail track": [230, 150, 140],
    "person": [220, 20, 60], "rider": [255, 0, 0],
    "car": [0, 0, 142], "truck": [0, 0, 70], "bus": [0, 60, 100], "on rails": [0, 80, 100], "motorcycle": [0, 0, 230],
    "bicycle": [119, 11, 32], "caravan": [0, 0, 90], "trailer": [0, 0, 110],
    "building": [70, 70, 70], "wall": [102, 102, 156], "fence": [190, 153, 153], "guard rail": [180, 165, 180],
    "bridge": [150, 100, 100], "tunnel": [150, 120, 90],
    "pole": [153, 153, 153], "pole group": [153, 153, 153], "traffic sign": [220, 220, 0], "traffic light": [250, 170, 30],
    "vegetation": [107, 142, 35], "terrain": [152, 251, 152],
    "sky": [70, 130, 180],
    "ground": [81, 0, 81], "dynamic": [111, 74, 0], "static": [81, 0, 21]
}

for filename in os.listdir(data_folder):
    if filename.endswith(".json"):
        json_path = os.path.join(data_folder, filename)

        with open(json_path, 'r') as f:
            data = json.load(f)

        for image_info in data['images']:
            image_file_name = image_info['file_name']
            image_path = os.path.join(data_folder, image_file_name)

            if os.path.exists(image_path):
                image = Image.open(image_path)

                fig, ax = plt.subplots(1, figsize=(10, 10))
                ax.imshow(image)

                category_mapping = {category['id']: category['name'] for category in data['categories']}

                for annotation in data['annotations']:
                    if annotation['image_id'] == image_info['id']:
                        segmentation = annotation['segmentation'][0]
                        category_id = annotation['category_id']
                        category_name = category_mapping[category_id]
                        poly_coords = [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]
                        color = colors.get(category_name, [0, 255, 0]) 
                        polygon = Polygon(poly_coords, closed=True, facecolor=np.array(color) / 255)
                        ax.add_patch(polygon)

                plt.axis('off')
                output_path = os.path.join(output_folder, f"visualized_{image_file_name}")
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

print("Segmentation visualizations with only colors completed and saved.")
