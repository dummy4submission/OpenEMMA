# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.
"""

from openemma.YOLO3D.script.Model import ResNet, ResNet18, VGG11
from openemma.YOLO3D.script import Model, ClassAverages
from openemma.YOLO3D.library.Plotting import *
from openemma.YOLO3D.library.Math import *
from openemma.YOLO3D.script.Dataset import generate_bins, DetectedObject
import numpy as np
from torchvision.models import resnet18
import torch.nn as nn
from openemma.YOLO3D.utils.torch_utils import select_device, time_sync
from openemma.YOLO3D.utils.general import (
    LOGGER,
    check_img_size,
    check_requirements,
    non_max_suppression,
    print_args,
    scale_coords,
)
from openemma.YOLO3D.utils.datasets import LoadImages
import argparse
import os
import sys
from pathlib import Path
import glob

import cv2
import torch
import subprocess
from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# model factory to choose model
model_factory = {
    "resnet": resnet18(pretrained=True),
    "resnet18": resnet18(pretrained=True),
    # 'vgg11': vgg11(pretrained=True)
}
regressor_factory = {"resnet": ResNet, "resnet18": ResNet18, "vgg11": VGG11}

colors = {
    "pedestrian": (180, 119, 31),  # Blue
    "trafficcone": (14, 127, 255),  # Orange
    "bicycle": (44, 160, 44),  # Green
    "bus": (40, 39, 214),  # Red
    "motocycle": (189, 103, 148),  # Purple
    "trailer": (75, 86, 140),  # Brown
    "truck": (75, 86, 140),  # Pink
    "car": (127, 127, 127),  # Gray
}


class Bbox:
    def __init__(self, box_2d, class_):
        self.box_2d = box_2d
        self.detected_class = class_


def detect3d(
    reg_weights,
    model_select,
    source,
    calib_file,
    show_result,
    save_result,
    output_path,
    roi_filter=None,
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    imgs_path = []
    if os.path.isfile(source):
        imgs_path = [source]
    else:
        # imgs_path = sorted(glob.glob(str(source) + '/*'))
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.tiff"):
            imgs_path.extend(glob.glob(os.path.join(source, ext)))

    calib = str(calib_file)

    # load model
    base_model = model_factory[model_select]
    regressor = regressor_factory[model_select](model=base_model).cuda()

    # load weight
    checkpoint = torch.load(reg_weights, weights_only=True)
    regressor.load_state_dict(checkpoint["model_state_dict"])
    regressor.eval()

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)

    # loop images
    for i, img_path in enumerate(imgs_path):
        # read image
        img = cv2.imread(img_path)
        img_name = os.path.basename(img_path)

        # Run detection 2d
        dets = detect2d(
            weights=os.path.join(current_dir, "yolo11n_nuimages.pt"),
            source=img_path,
            imgsz=[640, 640],
            device=0,
        )

        # for det in dets:
        #     cv2.rectangle(img, det.box_2d[0], det.box_2d[1], (255, 0, 255), 1)
        # cv2.imwrite(f'{output_path}/{i:03d}_2d.png', img)

        for det in dets:
            if not averages.recognized_class(det.detected_class):
                continue
            try:
                detectedObject = DetectedObject(
                    img, det.detected_class, det.box_2d, calib
                )
            except:
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = det.box_2d
            detected_class = det.detected_class

            input_tensor = torch.zeros([1, 3, 224, 224]).cuda()
            input_tensor[0, :, :, :] = input_img

            # predict orient, conf, and dim
            [orient, conf, dim] = regressor(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            location, X = calc_location(
                dim, proj_matrix, box_2d, alpha, theta_ray)
            orient = alpha + theta_ray

            # plot 3d detection
            if save_result or show_result:
                plot3d(
                    img,
                    proj_matrix,
                    det,
                    dim,
                    alpha,
                    theta_ray,
                    roi_filter=roi_filter,
                    img_2d=True,
                )
        imgs_output.append(img.copy())
        if show_result:
            cv2.imshow("3d detection", img)
            cv2.waitKey(0)

        if save_result and output_path is not None:
            try:
                os.mkdir(output_path)
            except:
                pass
            output_name = os.path.join(output_path, img_name)
            cv2.imwrite(f"{output_name}", img)


@torch.no_grad()
def detect2d(weights, source, imgsz, device, conf=0.5):

    # array for boundingbox detection
    bbox_list = []

    # Directories
    source = str(source)

    # Load model
    device = select_device(device)
    model = YOLO(weights)
    names = model.names
    stride = 32
    pt = True
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    dt, seen = [0.0, 0.0, 0.0], 0
    for _, im, im0s, _, _ in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, conf=conf)
        for r in pred:
            boxes = r.boxes
            for box in boxes:
                xyxy_ = torch.reshape(box.xyxy[0], (1, 4)).cpu().numpy()
                xyxy_ = scale_coords(im.shape[2:], xyxy_, im0s.shape).round()
                top_left, bottom_right = (int(xyxy_[0, 0]), int(xyxy_[0, 1])), (
                    int(xyxy_[0, 2]),
                    int(xyxy_[0, 3]),
                )
                bbox = [top_left, bottom_right]
                bbox_list.append(Bbox(bbox, names[int(box.cls[0])]))
    return bbox_list


def detect3DFromCVImg(
    reg_weights,
    model_select,
    imgs,
    calib_file,
    show_result,
    save_result,
    output_path,
    roi_filter=None,
):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # load model
    base_model = model_factory[model_select]
    regressor = regressor_factory[model_select](model=base_model).cuda()

    # load weight
    checkpoint = torch.load(reg_weights, weights_only=True)
    regressor.load_state_dict(checkpoint["model_state_dict"])
    regressor.eval()

    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)

    imgs_output = []
    if not isinstance(imgs,list):
        imgs = [imgs]
    # loop images
    for i, img in enumerate(imgs):
        img = img.copy()
        img_name = f"{i}.jpg"

        # Run detection 2d
        dets = detect2DFromCVImg(os.path.join(current_dir, "yolo11n_nuimages.pt"), img)

        # for det in dets:
        #     cv2.rectangle(img, det.box_2d[0], det.box_2d[1], (255, 0, 255), 1)
        # cv2.imwrite(f'{output_path}/{i:03d}_2d.png', img)

        for det in dets:
            if not averages.recognized_class(det.detected_class):
                continue
            try:
                detectedObject = DetectedObject(
                    img, det.detected_class, det.box_2d, calib_file
                )
            except Exception as e:
                print(f"An error occurred: {e}")
                continue

            theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img
            proj_matrix = detectedObject.proj_matrix
            box_2d = det.box_2d
            detected_class = det.detected_class

            input_tensor = torch.zeros([1, 3, 224, 224]).cuda()
            input_tensor[0, :, :, :] = input_img

            # predict orient, conf, and dim
            [orient, conf, dim] = regressor(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            location, X = calc_location(
                dim, proj_matrix, box_2d, alpha, theta_ray)
            orient = alpha + theta_ray

            plot3d(
                img,
                proj_matrix,
                det,
                dim,
                alpha,
                theta_ray,
                roi_filter=roi_filter,
                img_2d=True,
            )
        if show_result:
            cv2.imshow("3d detection", img)
            cv2.waitKey(0)

        if save_result and output_path is not None:
            try:
                os.mkdir(output_path)
            except:
                pass
            output_name = os.path.join(output_path, img_name)
            cv2.imwrite(f"{output_name}", img)

        imgs_output.append(img.copy())
    return imgs_output


def detect2DFromCVImg(weights, im, conf=0.5):

    # array for boundingbox detection
    bbox_list = []

    # Load model
    model = YOLO(weights)
    names = model.names

    # Inference
    pred = model(im, conf=conf)
    for r in pred:
        boxes = r.boxes
        for box in boxes:
            xyxy_ = torch.reshape(box.xyxy[0], (1, 4)).cpu().numpy()
            top_left, bottom_right = (int(xyxy_[0, 0]), int(xyxy_[0, 1])), (
                int(xyxy_[0, 2]),
                int(xyxy_[0, 3]),
            )
            bbox = [top_left, bottom_right]
            bbox_list.append(Bbox(bbox, names[int(box.cls[0])]))
    return bbox_list


def plot3d(
    img, proj_matrix, det, dimensions, alpha, theta_ray, img_2d=None, roi_filter=None
):
    box_2d = det.box_2d
    # the math! returns X, the corners used for constraint
    location, X = calc_location(
        dimensions, proj_matrix, box_2d, alpha, theta_ray)
    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    if roi_filter == None or roi_filter(theta_ray, location):
        color = colors[det.detected_class]
        plot_3d_box(
            img, proj_matrix, orient, dimensions, location, color=color, thickness=1
        )  # 3d boxes
    else:
        plot_3d_box(
            img,
            proj_matrix,
            orient,
            dimensions,
            location,
            color=(200, 200, 200),
            thickness=1,
        )  # 3d boxes

    return location


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default=ROOT / "yolov5s.pt",
        help="model path(s)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=ROOT / "eval/image_2",
        help="file/dir/URL/glob, 0 for webcam",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=ROOT / "data/coco128.yaml",
        help="(optional) dataset.yaml path",
    )
    parser.add_argument(
        "--imgsz",
        "--img",
        "--img-size",
        nargs="+",
        type=int,
        default=[640],
        help="inference size h,w",
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--classes",
        default=[0, 2, 3, 5],
        nargs="+",
        type=int,
        help="filter by class: --classes 0, or --classes 0 2 3",
    )
    parser.add_argument(
        "--reg_weights",
        type=str,
        default="weights/epoch_10.pkl",
        help="Regressor model weights",
    )
    parser.add_argument(
        "--model_select",
        type=str,
        default="resnet",
        help="Regressor model list: resnet, vgg, eff",
    )
    parser.add_argument(
        "--calib_file",
        type=str,
        default=ROOT / "eval/camera_cal/calib_cam_to_cam.txt",
        help="Calibration file or path",
    )
    parser.add_argument(
        "--show_result", action="store_true", help="Show Results with imshow"
    )
    parser.add_argument(
        "--save_result", action="store_true", help="Save result")
    parser.add_argument(
        "--output_path", type=str, default=ROOT / "output", help="Save output pat"
    )

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    detect3d(
        reg_weights=opt.reg_weights,
        model_select=opt.model_select,
        source=opt.source,
        calib_file=opt.calib_file,
        show_result=opt.show_result,
        save_result=opt.save_result,
        output_path=opt.output_path,
    )


def create_roi_filter(roi_r, roi_w, roi_d):
    def roi_filter(theta_ray, location):
        """_summary_

        Args:
            theta_ray (_type_): _description_
            location (list): Camera coordinate [right, down, front]

        Returns:
            _type_: _description_
        """
        obj_in_disk = False
        obj_in_lane = False
        if np.linalg.norm([location[0], location[2]]) < roi_r or roi_r < 0:
            obj_in_disk = True
        if (
            (
                np.abs(location[0]) < roi_w / 2.0
                and location[2] > 0
                and location[2] < roi_d
            )
            or roi_w < 0
            or roi_d < 0
        ):
            obj_in_lane = True
        return obj_in_disk or obj_in_lane

    return roi_filter


def yolo3d_nuScenes(img_path, output_path="", calib="nuscenes", save_result=False, roi_r=-1, roi_w=-1, roi_d=-1):
    """YOLO3D inference function for nuScenes dataset.

    Args:
        img_path (str): Path to an image or an image folder.
        roi_r (float): The radius of a disk-shaped region of interest.
        roi_w (float): The width of the region of interest in front of the vehicle.
        roi_d (float): The length of the region of interest in front of the vehicle.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Download weights
    if not os.path.isfile(os.path.join(current_dir, "weights", "resnet18.pkl")):
        print("Download weights...")
        script_dir = os.path.join(current_dir, "weights")
        script_path = os.path.join(script_dir, "get_weights.py")
        subprocess.run(["python", script_path, "--weights",
                       "resnet18"], cwd=script_dir)

    return detect3DFromCVImg(
        reg_weights=os.path.join(current_dir, "weights", "resnet18.pkl"),
        model_select="resnet18",
        imgs=img_path,
        calib_file=calib,
        show_result=False,
        save_result=save_result,
        output_path=output_path,
        roi_filter=create_roi_filter(roi_r, roi_w, roi_d),
    )

    # # Apply cam to ego translation for nuScenes
    # for bbox in bboxes:
    #     bbox[0] += 0.00475453292289
    #     bbox[1] += 1.72200568478
    #     bbox[2] += 1.49491291905

    # return bboxes, bboxes_roi


if __name__ == "__main__":
    # opt = parse_opt()
    # main(opt)
    yolo3d_nuScenes(
        "/home/cyqian/Open-EMMA-private/openemma/YOLO3D/eval/1",
        "/home/cyqian/Open-EMMA-private/openemma/YOLO3D/output/1",
        roi_r=20.0,
        roi_w=8.0,
        roi_d=40.0,
    )
