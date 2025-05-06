import os
import cv2
import csv
import numpy as np
from collections import defaultdict


def generate_masks_from_bboxes(csv_path, output_dir, image_shape=(224, 224)):
    """
    Reads bounding boxes from CSV and generates binary masks for each image.

    Args:
        csv_path (str): Path to bbox CSV file.
        output_dir (str): Where to save the mask images.
        image_shape (tuple): Shape (H, W) of the images to make blank masks.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Map image paths to a list of bboxes
    bbox_map = defaultdict(list)

    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            img_path, x1, y1, x2, y2, label = row
            bbox = list(map(float, [x1, y1, x2, y2]))
            bbox_map[img_path].append((bbox, label))

    print(f"Found {len(bbox_map)} unique images.")

    counter = 0
    for img_path, bbox_list in bbox_map.items():
        h, w = image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        for (bbox, label) in bbox_list:
            x1, y1, x2, y2 = map(int, bbox)
            # Draw filled white rectangle for the bbox
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Extract sub_path after 'rawframes/'
        sub_path = img_path.split("rawframes/")[1]  # 'malignant/malignant_9379001_1/000054.png'
        # Insert '_mask' before '.png'
        base, ext = os.path.splitext(sub_path)
        mask_path = os.path.join(output_dir, base + "_mask" + ext)
        # Dir output_dir + "/malignant/malignant_9379001_1/"
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv2.imwrite(mask_path, mask)

        # Output generation progress
        counter += 1
        if counter % 1000 == 0:
            print(f"Masks saved: {counter}/{len(bbox_map)}")


generate_masks_from_bboxes(
    csv_path="datasets/tsm_buv_imgs/predictions.csv",
    output_dir="../datasets/tsm_buv_imgs/masks",
    image_shape=(384, 450)  # adjust to real image size (h, w)
)
