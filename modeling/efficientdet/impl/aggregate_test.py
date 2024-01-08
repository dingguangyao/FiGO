# Core Author: Zylo117
# Script's Author: winter2897

"""
Simple Inference Script of EfficientDet-Pytorch for detecting objects on webcam
"""
import os
import time
import torch
import cv2
import numpy as np
# from torch.backends import cudnn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import (
    preprocess,
    invert_affine,
    postprocess,
    preprocess_video,
)

# Video's path
video_src = (
    "/home/gyding/data/ua_detrac/ua_detrac.mp4"  # set int to use webcam, set str to read from a video file
)

compound_coef = 6

threshold = 0.2
iou_threshold = 0.2

use_cuda = True

label_list = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "",
    "backpack",
    "umbrella",
    "",
    "",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "",
    "dining table",
    "",
    "",
    "toilet",
    "",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
input_size = (
    input_sizes[compound_coef]
)

# load model
print("load model...")
model = EfficientDetBackbone(
    compound_coef=compound_coef, 
    num_classes=len(label_list),
    ratios=anchor_ratios,
    scales=anchor_scales,
)
model.load_state_dict(torch.load(f"/home/gyding/FiGO/weights/efficientdet/efficientdet-d{compound_coef}.pth"))
model.requires_grad_(False)
model.eval()

if use_cuda:
    model = model.cuda()

# function for display
# def display(preds, imgs):
#     for i in range(len(imgs)):
#         if len(preds[i]["rois"]) == 0:
#             return imgs[i]

#         for j in range(len(preds[i]["rois"])):
#             (x1, y1, x2, y2) = preds[i]["rois"][j].astype(np.int)
#             cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
#             obj = label_list[preds[i]["class_ids"][j]]
#             score = float(preds[i]["scores"][j])

#             cv2.putText(
#                 imgs[i],
#                 "{}, {:.3f}".format(obj, score),
#                 (x1, y1 + 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (255, 255, 0),
#                 1,
#             )

#         return imgs[i]

def out_to_std_out(out):
    cls = [int(c) for c in out[0]["class_ids"]]
    score = [float(s) for s in out[0]["scores"]]
    return {"class": cls, "score": score}

def cat_id_to_label(idx):
    if idx >= len(label_list):
        raise Exception("EfficeintDet category id does not exist.")
    return label_list[idx]

# Box
regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

# Video capture
# cap = cv2.VideoCapture(video_src)
root_dir = os.path.join("/home/gyding/data/ua_detrac", 'MVI_40981')
length = len(os.listdir(root_dir))
img_path_list = os.listdir(root_dir)

print("inference...")
per_car_count = 0
per_image = []
car_count = 0
frame_count = 0
st = time.perf_counter()

for i in range(length):
    per_car_count = 0
    img_path = os.path.join(root_dir, img_path_list[i])

    # frame preprocessing
    ori_imgs, framed_imgs, framed_metas = preprocess(
        img_path, max_size=input_size
    )

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.permute(
        0, 3, 1, 2
    )

    # model predict
    with torch.no_grad():
        features, regression, classification, anchors = model(x)

        out = postprocess(
            x,
            anchors,
            regression,
            classification,
            regressBoxes,
            clipBoxes,
            threshold,
            iou_threshold,
        )

    # result
    out = invert_affine(framed_metas, out)

    out = out_to_std_out(out)

    for k in range(len(out["class"])):
            out["class"][k] = cat_id_to_label(out["class"][k])

    # print(out)
    frame_count += 1
    for k, c in enumerate(out['class']):
            if c == 'car' and out['score'][k] >= 0.5:
                 car_count += 1
                 per_car_count += 1
    
    per_image.append(per_car_count)

exec_time = time.perf_counter() - st

print(f"d{compound_coef}", frame_count, car_count, car_count/frame_count)
print(exec_time)
print(per_image)
# cap.release()
# cv2.destroyAllWindows()
