# The new config inherits a base config to highlight the necessary modification
_base_ = "../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2)
        #mask_head=dict(num_classes=2)
        ))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('cat', 'dog',)
data = dict(
    train=dict(
        img_prefix='dataset/catdog/train/',
        classes=classes,
        ann_file='dataset/catdog/train/annotation_coco.json'),
    val=dict(
        img_prefix='dataset/catdog/val/',
        classes=classes,
        ann_file='dataset/catdog/val/annotation_coco.json'),
    test=dict(
        img_prefix='dataset/catdog/test/',
        classes=classes,
        ann_file='dataset/catdog/test/annotation_coco.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'