import os
import csv
import json


def convert_json_to_csv(json_path, images_base_path, annotations_csv_path, classes_csv_path):
    # Load JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Map category id to name [1: "benign", 2: "malignant"]
    categories = {category['id']: category['name'] for category in data['categories']}

    # Map image id to file_name [1: "benign/x28f299ceb056964c/000000.png"]
    images = {image['id']: image['file_name'] for image in data['images']}

    # Prepare annotation rows
    annotation_rows = []
    for annotation in data['annotations']:
        image_id = annotation['image_id']

        if image_id not in images:
            print(f"Warning: image_id {image_id} not found in images. Skipping.")
            continue

        file_path = os.path.join(images_base_path, images[image_id])

        x1, y1, w, h = annotation['bbox']  # bounding box data
        x2 = x1 + w
        y2 = y1 + h

        # Make x1, y1 always the min; x2, y2 always the max
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        if x1 == x2 or y1 == y2: continue

        category_name = categories[annotation['category_id']]

        annotation_rows.append([file_path, int(x1), int(y1), int(x2), int(y2), category_name])

    # Write annotations.csv
    with open(annotations_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in annotation_rows:
            writer.writerow(row)

    # Write classes.csv
    with open(classes_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for category_id, category_name in categories.items():
            writer.writerow([category_name, category_id])

    print(f'Done! Annotations written to {annotations_csv_path}, Classes written to {classes_csv_path}.')


dataset_path = "datasets/miccai_2022_buv_imgs/"

# Example usage
convert_json_to_csv(
    json_path=dataset_path+'imagenet_vid_train_15frames.json',
    images_base_path=dataset_path+'rawframes/',
    annotations_csv_path=dataset_path+'annotations_train.csv',
    classes_csv_path=dataset_path+'classes_train.csv'
)
