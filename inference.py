import torch
from matplotlib import pyplot as plt
from mmdet.apis import init_detector, inference_detector
import argparse
import os

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

    for root, dirs, files in os.walk(args.image_folder_path):
        for file in files:
            img_path = os.path.join(root, file)
            print(file)
            # inference the demo image
            detections = inference_detector(model, img_path)
            print(detections)
            name, type = file.split('.')[0], file.split('.')[1]
            output_path = args.output_path + '/' + name + "_res" + '.' + type
            model.show_result(img_path, detections, out_file=output_path)

if __name__ == '__main__':
    main()
