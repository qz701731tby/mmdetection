import torch
from matplotlib import pyplot as plt
from mmdet.apis import init_detector, inference_detector
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
#device='cuda:0'
# init a detector
model = init_detector(config_file, checkpoint_file, device=device)
# inference the demo image
img_path = 'image/test2.jpg'
output_path = 'output/res2.jpg'
detections = inference_detector(model, img_path)
model.show_result(img_path, detections, out_file=output_path)
print(detections)
#plt.imshow(img)