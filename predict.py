import denoise_micrographs
from glob import glob
import pandas as pd
import os
import csv
import cv2
import sys
import random
import argparse
from pathlib import Path
from typing import Iterable
from PIL import Image
import numpy as np

import torch

import util.misc as utils

from models import build_model
from datasets.micrograph import make_micrograph_transforms

import matplotlib.pyplot as plt
import time


def nms(bounding_boxes, confidence_scores, nms_threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_scores)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_scores[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < nms_threshold)
        order = order[left]

    picked_boxes = np.array(picked_boxes).squeeze()
    picked_score = np.array(picked_score)

    return picked_boxes, picked_score

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h,
                            img_w, img_h
                            ], dtype=torch.float32)
    return b

#changes by Ashwin
def get_images(in_path):  
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm' or ext == '.mrc':
                img_files.append(os.path.join(dirpath, file))

    return img_files


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    # Test hyperparameters
    parser.add_argument('--quartile_threshold', type=float, default=0.25, help='Quartile threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.7, help='Non-maximum suppression threshold')
    parser.add_argument('--empiar', default='10005', help='EMPIAR ID for prediction')
    parser.add_argument('--remarks', default='CryoTransformer_predictions', help='Additional remarks')
    parser.add_argument('--du_particles', default='N', choices=['Y', 'N'], help='DU Particles (Y or N)')
    parser.add_argument('--num_queries', type=int, default=600, help='Number of queries')
    parser.add_argument('--save_micrographs_with_encircled_proteins', default='Y', choices=['Y', 'N'], help='Plot predicted proteins on Micrographs (Y or N)')
    parser.add_argument('--resume', default='/bml/ashwin/ViTPicker/particle_picker/model_OBackbone_DETR/output/backboneresnet152_dataset22_updated_Denoised_Datasets_num_queries600_batch16_epoch300_remarksyes_pretrained_weights_timestamp_2023-07-29 17:20:16/checkpoint0299.pth', help='Resume path')
  

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet152', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='micrograph')
    parser.add_argument('--data_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')


    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--thresh', default=0, type=float)   #edits by Ashwin, initially 0.5

    return parser


@torch.no_grad()
def infer(images_path, model, postprocessors, device, output_dir):
    model.eval()
    duration = 0

    prefix_file_name = "{}_DU_{}_Qthres{}_NMSthres{}_remarks_{}".format(
    args.empiar, args.du_particles, args.quartile_threshold, args.nms_threshold, args.remarks
    )

    for img_sample in images_path:
        filename = os.path.basename(img_sample)[:-4] 
        print(len(filename))
        extension = img_sample[-3:]
        #loading image if input is in jpg format
        if extension == 'jpg':
            orig_image = Image.open(img_sample)
            img_size = orig_image.size
            rgb_image = Image.new("RGB", img_size)
            rgb_image.putdata([(x,x,x) for x in orig_image.getdata()])
            w, h = rgb_image.size

        if extension == 'mrc':
            orig_image = denoise_micrographs.denoise(img_sample)
            h, w = orig_image.shape
            # Create a new 3D array with shape (height, width, 3)
            rgb_array = np.zeros((h, w, 3), dtype=np.uint8)
            # Set the same intensity value for all three color channels at each pixel location
            for y in range(h):
                for x in range(w):
                    intensity = orig_image[y, x]
                    rgb_array[y, x, 0] = intensity  # Red channel
                    rgb_array[y, x, 1] = intensity  # Green channel
                    rgb_array[y, x, 2] = intensity  # Blue channel
            # Convert the NumPy array to a PIL Image
            rgb_image = Image.fromarray(rgb_array)

        transform = make_micrograph_transforms("val")
        dummy_target = {
            "size": torch.as_tensor([int(h), int(w)]),
            "orig_size": torch.as_tensor([int(h), int(w)])
        }
        image, targets = transform(rgb_image, dummy_target)
        image = image.unsqueeze(0)
        image = image.to(device)


        conv_features, enc_attn_weights, dec_attn_weights = [], [], []
        hooks = [
            model.backbone[-2].register_forward_hook(
                        lambda self, input, output: conv_features.append(output)

            ),
            model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                        lambda self, input, output: enc_attn_weights.append(output[1])

            ),
            model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_attn_weights.append(output[1])

            ),

        ]

        start_t = time.perf_counter()
        outputs = model(image)
        end_t = time.perf_counter()
        outputs["pred_logits"] = outputs["pred_logits"].cpu()
        # print(outputs["pred_logits"])
        outputs["pred_boxes"] = outputs["pred_boxes"].cpu()
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        # print("=============probas softmax ===============================")
        # print(probas)

        probas2 = outputs['pred_logits'].sigmoid()
        topk_values, topk_indexes = torch.topk(probas2.view(outputs["pred_logits"].shape[0], -1), args.num_queries, dim=1)   #extreme important mention num queries
        scores = topk_values
        keep = scores[0] > np.quantile(scores, args.quartile_threshold)  #This is what prevents from predicting ice patches as particles
        scores = scores[0, keep]


        # keep = probas.max(-1).values > args.thresh  #this is original
        # print("==========" + img_sample + "====pred_logits after softmax===============================")
        # print(keep )

        bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], rgb_image.size)
        probas = probas[keep].cpu().data.numpy()


        for hook in hooks:
            hook.remove()

        conv_features = conv_features[0]
        enc_attn_weights = enc_attn_weights[0]
        dec_attn_weights = dec_attn_weights[0].cpu()

        # get the feature map shape
        # h, w = conv_features['0'].tensors.shape[-2:]
        scores = scores.cpu().detach().numpy()
        boxes, scores = nms(bboxes_scaled, scores, nms_threshold=args.nms_threshold)
        print(f"----- generating star file for {filename}")
        # create directory for star files if not exist:
        box_file_path = output_dir + '/box_files/'
        bounding_box_images_path = output_dir + 'predicted_bounding_box_images/'
        if not os.path.exists(box_file_path):
            os.makedirs(box_file_path)
        save_individual_box_file(boxes, scores, img_sample, h, box_file_path, "_vitpicker")
        # print("=============boxes  ===============================")
        # print(boxes)
        # print("=============scores  ===============================")
        # print(scores)
        #edits by Ashwin
        if len(bboxes_scaled) == 0:
            print("there are no particle in image")
            continue

        if args.save_micrographs_with_encircled_proteins == 'Y':
            plot_predicted_boxes(rgb_image, boxes, filename, bounding_box_images_path, h)

        # print("=============== Predictions saved ===================")
        # cv2.imshow("img", img)
        # cv2.waitKey()
        # infer_time = end_t - start_t
        # duration += infer_time
        # print("Processing END...{} ({:.3f}s)".format(filename, infer_time))

    # avg_duration = duration / len(images_path)

    # print("Avg. Time: {:.3f}s".format(avg_duration))

    #making header for combined star file:
    save_combined_star_file(box_file_path, prefix_file_name)


def save_individual_box_file(boxes, scores, img_file, h, box_file_path, out_imgname):
    write_name = box_file_path + os.path.basename(img_file)[:-4] + out_imgname + '.box'
    with open(write_name, "w") as boxfile:
        boxwriter = csv.writer(
            boxfile, delimiter='\t', quotechar="|", quoting=csv.QUOTE_NONE
        )
        boxwriter.writerow(["Micrograph_Name    X_Coordinate    Y_Coordinate    Class_Number    AnglePsi    Confidence_Score"])

        for i, box in enumerate(boxes):
            star_bbox = box.cpu().data.numpy()
            star_bbox = star_bbox.astype(np.int32)
            #h- is done to handle the cryoSparc micrograph reading orientation
            boxwriter.writerow([os.path.basename(img_file)[:-4] + '.mrc', (star_bbox[0] + star_bbox[2]) / 2, h-(star_bbox[1] + star_bbox[3]) / 2, -9999, -9999, scores[i]])
            if args.du_particles == 'Y':
                coordinate_shift_rand = random.choice(list(range(-20, -9)) + list(range(10, 21))) #shifting center to obtain better 2D averaging
                # coordinate_shift_rand = 10
                boxwriter.writerow([os.path.basename(img_file)[:-4] + '.mrc', ((star_bbox[0] + star_bbox[2]) / 2)+coordinate_shift_rand, (h-(star_bbox[1] + star_bbox[3]) / 2)+coordinate_shift_rand, -9999, -9999, scores[i]])

def plot_predicted_boxes(rgb_image, boxes, filename, bounding_box_images_path, h):
    img = np.array(rgb_image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for idx, box in enumerate(boxes):
        bbox = box.cpu().data.numpy()
        bbox = bbox.astype(np.int32)
        bbox_d = bbox.astype(np.int32)
        bbox_circle = bbox.astype(np.int32)


        bbox = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
            ])
        bbox = bbox.reshape((4, 2))
        # bbox_d = np.array([
        #     [bbox_d[0]+15, bbox_d[1]+15],
        #     [bbox_d[2]+15, bbox_d[1]+15],
        #     [bbox_d[2]+15, bbox_d[3]+15],
        #     [bbox_d[0]+15, bbox_d[3]+15],
        #     ])
        # bbox_d = bbox_d.reshape((4, 2))


        bbox_circle_center = np.array([(bbox_circle[0] + bbox_circle[2]) / 2, (bbox_circle[1] + bbox_circle[3])/2]) #h- is ommitted here to handle the image plot
        bbox_circle_center = bbox_circle_center.reshape((1, 2))

        x_coordinate, y_coordinate = bbox_circle_center[0]
        center = (int(x_coordinate), int(y_coordinate))


        # cv2.polylines(img, [bbox], True, (0, 255, 0), 4)
        # color=(0,255,0) #green
        color =(150, 255, 255) #purple
        radius=81
        thickness=10 # 7 earlier
        # cv2.polylines(img, [bbox_d], True, (0, 255, 0), 4)
        cv2.circle(img, center, radius, color, thickness)

    img_save_path = os.path.join(output_dir, f"{filename}.jpg")

    cv2.imwrite(img_save_path, img)


def save_combined_star_file(box_file_path, prefix_file_name):
    text_files = [file for file in os.listdir(box_file_path) if file.endswith('.box')]
    text_files.sort()
    output_file = output_dir + prefix_file_name + '_' + 'combined_star_file.star'
    header = '''
data_

loop_
_rlnMicrographName #1 
_rlnCoordinateX #2 
_rlnCoordinateY #3 
_rlnClassNumber #4 
_rlnAnglePsi #5
_rlnAutopickFigureOfMerit #6
'''

    with open(output_file, 'w') as outfile:
        # Write the header content to the new file
        outfile.write(header)

        # Iterate over each text file
        for file in text_files:
            # Open the current file in read mode
            with open(os.path.join(box_file_path, file), 'r') as infile:
                # Skip the first line
                next(infile)
                # Read the remaining content of the file
                content = infile.read()
                # Write the content to the new file
                outfile.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('CryoTransformer training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    from datetime import datetime
    current_datetime = datetime.now()
    timestamp = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    # data_path = "/bml/Rajan_CryoEM/Processed_Datasets/test_dataset/{}/images".format(args.empiar) #which data to predict: if 300?
    data_path = "/bml/ashwin/ViTPicker/25_test_mrc_for_cryosparck_load/{}".format(args.empiar) #which data to predict: if 25?
    output_dir = "output/predictions_EMPIAR_{}_DU_{}_predictions_thres{}_nms_thres{}_num_queries{}_remarks_{}_timestamp_{}/".format(
    args.empiar, args.du_particles, args.quartile_threshold, args.nms_threshold, args.num_queries, args.remarks, timestamp)


    Path(output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model, _, postprocessors = build_model(args)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device)
    image_paths = get_images(data_path)
    print(image_paths)

    infer(image_paths, model, postprocessors, device, output_dir)