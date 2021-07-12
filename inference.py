import torch
from matplotlib import pyplot as plt
from mmdet.apis import init_detector, inference_detector
import argparse
import os
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='set config file path')
    parser.add_argument('-p', '--checkpoint', help='set checkpoint path')
    parser.add_argument('-i', '--image_folder_path', help='set image folder path')
    parser.add_argument('-o', '--output_path', help='set output path')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # init a detector
    model = init_detector(args.config, args.checkpoint, device=device)

    images = glob.glob(os.path.join(args.image_folder_path, "*.jpeg")) + glob.glob(os.path.join(args.image_folder_path, "*.jpg"))
    print(images)
    for image in images:
        detections = inference_detector(model, image)
        name = image.split('/')[-1].split('.')[0]
        output_path = args.output_path + '/' + name + "_res" + ".jpg"
        model.show_result(image, detections, out_file=output_path)

if __name__ == '__main__':
    main()
