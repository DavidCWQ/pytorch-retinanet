import argparse

import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval

# ------------------ Check Torch Version -------------------
# assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple evaluation script for a trained RetinaNet model.')

    # ------------------ Argument Parsing ------------------
    parser.add_argument('--csv_annotations_path', help='Path to CSV annotations')
    parser.add_argument('--model_path', help='Path to model', type=str)
    parser.add_argument('--images_path', help='Path to images directory', type=str)
    parser.add_argument('--class_list_path', help='Path to class list CSV', type=str)
    parser.add_argument('--iou_threshold', help='IOU threshold used for evaluation', type=str, default='0.5')

    parser = parser.parse_args(args)

    # ---------------- Dataset Preparation -----------------
    dataset_val = CSVDataset(
        parser.csv_annotations_path,
        parser.class_list_path,
        transform=transforms.Compose([
            Normalizer(),
            Resizer()
        ])
    )

    # ------------------- Model Loading --------------------
    retinanet = torch.load(parser.model_path, weights_only=False)

    use_gpu = True

    if use_gpu and torch.cuda.is_available():
        retinanet = retinanet.cuda()

    # Wrap model with DataParallel for multi-GPU support
    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    # --------------- Model Evaluation Mode ----------------
    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    # --------------------- Evaluation ---------------------
    print(csv_eval.evaluate(dataset_val, retinanet, iou_threshold=float(parser.iou_threshold)))


if __name__ == '__main__':
    main()
