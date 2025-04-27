import argparse

import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet import coco_eval

# ------------------ Check Torch Version -------------------
assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple evaluation script for RetinaNet on COCO dataset.')

    # ------------------ Argument Parsing ------------------
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model', type=str)

    parser = parser.parse_args(args)

    # ---------------- Dataset Preparation -----------------
    dataset_val = CocoDataset(
        parser.coco_path,
        set_name='val2017',
        transform=transforms.Compose([
            Normalizer(),
            Resizer()
        ])
    )

    # ------------------- Model Creation -------------------
    retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)

    use_gpu = True

    if use_gpu and torch.cuda.is_available():
        retinanet = retinanet.cuda()

    # ------------------- Model Loading --------------------
    if torch.cuda.is_available():
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    # --------------- Model Evaluation Mode ----------------
    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    # --------------------- Evaluation ---------------------
    coco_eval.evaluate_coco(dataset_val, retinanet)


if __name__ == '__main__':
    main()
