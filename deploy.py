import os
import csv
import torch
import skimage
import numpy as np

from torchvision import transforms
from retinanet.dataloader import InferenceNormalizer, InferenceResizer

# Resize the image, standardizes the image pixel values
transform = transforms.Compose([
    InferenceNormalizer(),
    InferenceResizer()
])

def load_model(_model_path):
    retinanet = torch.load(_model_path, weights_only=False)
    # ------- Wrap with DataParallel for GPU support -------
    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
        print('CUDA available: {}'.format(torch.cuda.is_available()))
    else:
        print('CUDA available: {}. Run on CPU.'.format(torch.cuda.is_available()))
    # --------------- Model Evaluation Mode ----------------
    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()
    # ------------------- Return Model ---------------------
    return retinanet

def load_images_from_tsm_path(_folder, _count):
    images = []
    for i in range(_count):
        img_path = os.path.join(_folder, f"{i:06}.png")
        img = skimage.io.imread(img_path)
        # Convert RGBA to RGB by removing the alpha channel (if the image is RGBA)
        if img.shape[-1] == 4:  # Check if the image has 4 channels (RGBA)
            img = img[..., :3]  # Take only the first 3 channels (RGB)
        if len(img.shape) == 2: # If the image is grayscale, convert it to RGB
            img = skimage.color.gray2rgb(img)
        img = img.astype(np.float32) / 255.0
        images.append(img)
    return images

def detect_objects(model, images):
    boxes_per_image = []

    for img in images:
        sample = {'img': np.array(img)}
        sample = transform(sample)
        scale = sample['scale']
        img = sample['img']

        with torch.no_grad():
            scores, classifications, boxes = model(
                img.permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
            )

        scores = scores.cpu().numpy()
        boxes = boxes.cpu().numpy()
        labels = classifications.cpu().numpy()

        boxes /= scale
        max_detections = 10
        score_threshold = 0.5

        # Filter out low-confidence predictions
        indices = np.where(scores > score_threshold)[0]

        if indices.shape[0] > 0:
            scores = scores[indices]
            boxes = boxes[indices]
            labels = labels[indices]

            # Sort scores in descending order and take top-k
            scores_sort = np.argsort(-scores)[:max_detections]

            # Keep top-k predictions
            scores = scores[scores_sort]
            boxes = boxes[scores_sort]
            labels = labels[scores_sort]

            predictions = []
            for i in range(len(scores)):
                predictions.append({
                    'bbox': boxes[i].tolist(),
                    'score': float(scores[i]),
                    'label': int(labels[i])
                })
            boxes_per_image.append(predictions)
        else:
            boxes_per_image.append([])

    return boxes_per_image

def main(_model_path, _txt_path, _output_path):
    model = load_model(_model_path)
    label_name = {0: "benign", 1: "malignant"}

    try:
        with open(_txt_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at path {_txt_path}.")
        exit(-1)
    except IOError as e:
        print(f"Error: while reading file {_txt_path}: {e}")
        exit(-1)
    print(f"Txt File Loaded.")

    with open(_output_path, mode='w', newline='') as csvfile:  # Moved here
        writer = csv.writer(csvfile)

        for line in lines:
            path, count, label = line.strip().split()
            count = int(count)
            label = int(label)

            print(f"Processing: {path}, {count} images, class: {label_name[label]}")
            images = load_images_from_tsm_path(path, count)
            bboxes = detect_objects(model, images)

            for idx, predicts in enumerate(bboxes):
                image_name = f"{idx:06}.png"
                image_path = os.path.join(path, image_name)

                for p in predicts:
                    bbox = p['bbox']  # Assumes bbox is [x1, y1, x2, y2]
                    writer.writerow([image_path] + bbox + [label_name[p['label']-1]]) # Bug to fix bug


if __name__ == "__main__":
    model_path = 'datasets/miccai_2022_buv_imgs/models/csv_retinanet_59.pt'
    txt_path = 'datasets/tsm_buv_imgs/my_tsm_train_paths.txt'
    csv_path = 'datasets/tsm_buv_imgs/predictions.csv'
    main(model_path, txt_path, csv_path)
