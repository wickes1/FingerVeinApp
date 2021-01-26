import math
from PIL import Image
import cv2
from skimage import exposure
from scipy.ndimage import convolve
import numpy as np
import numpy
import os
import torch
from .resnet import resnet18
from torchvision import transforms
from sklearn import preprocessing
import pickle
import argparse
from collections import OrderedDict


def imfilter(a, b, gpu=False, conv=True):
    """imfilter function based on MATLAB implementation."""
    if a.dtype == np.uint8:
        a = a.astype(np.float64) / 255  # convert to a numpy float64
    M, N = a.shape
    if conv == True:
        b = np.rot90(b, k=2)  # rotate the image by 180 degree
        return convolve(a, b, mode="nearest")


class Finger_crop:
    def __init__(
        self,
        mask_h=4,  # Height of the mask
        mask_w=20,  # Width of the mask
        heq=False,
        padding_offset=5,  # Always the same
        padding_threshold=0.2,  # 0 for UTFVP database (high quality), 0.2 for VERA database (low quality)
        gpu=False,
        color_channel="gray",  # the color channel to extract from colored images, if colored images are in the database
        output_w=144,
        output_h=64,
        dataset="SDMULA",
        **kwargs
    ):  # parameters to be written in the __str__ method

        self.mask_h = mask_h
        self.mask_w = mask_w
        self.heq = heq
        self.padding_offset = padding_offset
        self.padding_threshold = padding_threshold
        self.gpu = gpu
        self.color_channel = color_channel
        self.output_w = output_w
        self.output_h = output_h
        self.dataset = dataset

    def __correct_edge__(self, edge, position, thred=5):
        edge_diff = np.array([s - t for s, t in zip(edge, edge[1:])])
        if position == "up":
            ind_1 = np.where(edge_diff < -1 * thred)[0]
            ind_2 = np.where(edge_diff > thred)[0]
            if len(ind_1) > 0:
                for ind in ind_1[::-1]:
                    # edge[0:ind_1[-1] + 1] = edge[ind_1[-1] + 1]
                    if ind < len(edge) * 0.8:
                        edge[0 : ind + 1] = edge[ind + 1]
                        break
            if len(ind_2) > 0:
                for ind in ind_2:
                    if ind > len(edge) * 0.2:
                        edge[ind + 1 :] = edge[ind]
                        break
                # edge[ind_2[0] + 1:] = edge[ind_2[0]]
        elif position == "down":
            ind_1 = np.where(edge_diff > thred)[0]
            ind_2 = np.where(edge_diff < -1 * thred)[0]
            if len(ind_1) > 0:
                for ind in ind_1[::-1]:
                    # edge[0:ind_1[-1] + 1] = edge[ind_1[-1] + 1]
                    if ind < len(edge) * 0.8:
                        edge[0 : ind + 1] = edge[ind + 1]
                        break
            if len(ind_2) > 0:
                for ind in ind_2:
                    if ind > len(edge) * 0.2:
                        edge[ind + 1 :] = edge[ind]
                        break
                # edge[ind_2[0] + 1:] = edge[ind_2[0]]
        return edge

    def __leemask__(self, image):
        img_h, img_w = image.shape

        # Determine lower half starting point vertically
        if numpy.mod(img_h, 2) == 0:
            half_img_h = img_h // 2 + 1
        else:
            half_img_h = numpy.ceil(img_h / 2)

        # Determine lower half starting point horizontally
        if numpy.mod(img_w, 2) == 0:
            half_img_w = img_w // 2 + 1
        else:
            half_img_w = numpy.ceil(img_w / 2)

        # Construct mask for filtering
        mask = numpy.zeros((self.mask_h, self.mask_w))
        mask[0 : self.mask_h // 2, :] = -1
        mask[self.mask_h // 2 :, :] = 1

        img_filt = imfilter(image, mask, self.gpu, conv=True)
        # Upper part of filtred image
        img_filt_up = img_filt[0 : half_img_h - 1, :]
        y_up = img_filt_up.argmax(axis=0)
        # for SDMULA and FVUSM, no need for MMCBNU_6000, some images of MMCBNU_6000 need to be manually correct edge
        # y_up = self.__correct_edge__(y_up, position='up')

        # Lower part of filtred image
        img_filt_lo = img_filt[half_img_h - 1 :, :]
        y_lo = img_filt_lo.argmin(axis=0)
        # for SDMULA and FVUSM, no need for MMCBNU_6000, some images of MMCBNU_6000 need to be manually correct edge
        # y_lo = self.__correct_edge__(y_lo, position='down')

        img_filt = imfilter(image, mask.T, self.gpu, conv=True)
        # Left part of filtered image
        # img_filt_lf = img_filt[:, 0:half_img_w]
        # y_lf = img_filt_lf.argmax(axis=1)

        # Right part of filtred image
        img_filt_rg = img_filt[:, half_img_w:]
        y_rg = img_filt_rg.argmin(axis=1)

        finger_mask = np.zeros(image.shape, dtype=np.bool)
        for i in range(0, y_up.size):
            finger_mask[y_up[i] : y_lo[i] + img_filt_lo.shape[0] + 1, i] = True

        # Left region
        # for i in range(0, y_lf.size):
        #     finger_mask[i, 0:y_lf[i] + 1] = False
        # for i in range(0, y_lf.size):
        #     finger_mask[:, 0:int(numpy.median(y_lf[i]))] = False

        # Right region has always the finger ending, crop the padding with the meadian
        if self.dataset == "FVUSM":
            finger_mask[:, int(numpy.median(y_rg)) + img_filt_rg.shape[1] :] = False

        # Extract y-position of finger edges
        edges = numpy.zeros((2, img_w))
        edges[0, :] = y_up
        # edges[0, 0:int(round(numpy.mean(y_lf))) + 1] = edges[0, int(round(numpy.mean(y_lf))) + 1]

        edges[1, :] = y_lo + img_filt_lo.shape[0]
        # edges[1, 0:int(round(numpy.mean(y_lf))) + 1] = edges[1, int(round(numpy.mean(y_lf))) + 1]

        return (finger_mask, edges)

    def __leemaskMATLAB__(self, image):
        img_h, img_w = image.shape

        # Determine lower half starting point
        if numpy.mod(img_h, 2) == 0:
            half_img_h = img_h // 2 + 1
        else:
            half_img_h = numpy.ceil(img_h / 2)

        # Construct mask for filtering
        mask = numpy.zeros((self.mask_h, self.mask_w))
        mask[0 : self.mask_h // 2, :] = -1
        mask[self.mask_h // 2 :, :] = 1

        img_filt = imfilter(image, mask, self.gpu, conv=True)

        # Upper part of filtred image
        img_filt_up = img_filt[0 : img_h // 2, :]
        y_up = img_filt_up.argmax(axis=0)

        # Lower part of filtred image
        img_filt_lo = img_filt[half_img_h - 1 :, :]
        y_lo = img_filt_lo.argmin(axis=0)

        for i in range(0, y_up.size):
            img_filt[y_up[i] : y_lo[i] + img_filt_lo.shape[0], i] = 1

        finger_mask = numpy.ndarray(image.shape, numpy.bool)
        finger_mask[:, :] = False

        finger_mask[img_filt == 1] = True

        # Extract y-position of finger edges
        edges = numpy.zeros((2, img_w))
        edges[0, :] = y_up
        edges[1, :] = numpy.round(y_lo + img_filt_lo.shape[0])

        return (finger_mask, edges)

    def __huangnormalization__(self, image, mask, edges):
        img_h, img_w = image.shape

        bl = (edges[0, :] + edges[1, :]) / 2  # Finger base line
        x = numpy.arange(0, img_w)
        A = numpy.vstack([x, numpy.ones(len(x))]).T

        # Fit a straight line through the base line points
        w = numpy.linalg.lstsq(A, bl, rcond=None)[0]  # obtaining the parameters

        angle = -1 * math.atan(w[0])  # Rotation
        tr = img_h / 2 - w[1]  # Translation
        scale = 1.0  # Scale

        # Affine transformation parameters
        sx = sy = scale
        cosine = math.cos(angle)
        sine = math.sin(angle)

        a = cosine / sx
        b = -sine / sy
        # b = sine/sx
        c = 0  # Translation in x

        d = sine / sx
        e = cosine / sy
        f = tr  # Translation in y
        # d = -sine/sy
        # e = cosine/sy
        # f = 0

        g = 0
        h = 0
        # h=tr
        i = 1

        T = numpy.matrix([[a, b, c], [d, e, f], [g, h, i]])
        Tinv = numpy.linalg.inv(T)
        Tinvtuple = (
            Tinv[0, 0],
            Tinv[0, 1],
            Tinv[0, 2],
            Tinv[1, 0],
            Tinv[1, 1],
            Tinv[1, 2],
        )

        img = Image.fromarray(image)
        image_norm = img.transform(
            img.size, Image.AFFINE, Tinvtuple, resample=Image.BICUBIC
        )
  
        image_norm = numpy.array(image_norm)

        finger_mask = numpy.zeros(mask.shape)
        finger_mask[mask == True] = 1

        img_mask = Image.fromarray(finger_mask)
        mask_norm = img_mask.transform(
            img_mask.size, Image.AFFINE, Tinvtuple, resample=Image.BICUBIC
        )

        mask_norm = numpy.array(mask_norm)

        mask[:, :] = False
        mask[mask_norm == 1] = True

        return (image_norm, mask)


    def crop_finger(self, image):

        if self.heq:
            image = exposure.equalize_hist(image)
        else:
            image = image

        finger_mask, finger_edges = self.__leemask__(image)
        ori_mask = (finger_mask * 255).astype("uint8")
        image_norm, finger_mask_norm = self.__huangnormalization__(
            image, finger_mask, finger_edges
        )

        mask = (finger_mask_norm * 255).astype(np.uint8)
        rect = cv2.boundingRect(mask)
        x, y, w, h = rect[0], rect[1], rect[2], rect[3]
        output = image_norm[y : y + h, x : x + w]
        output = cv2.resize(output, (self.output_w, self.output_h))

        return output, finger_edges


class FVR:
    def __init__(self, model_path="./model/snapshot.pth", threshold=0.75):
        snapshot = torch.load(model_path, map_location=torch.device("cpu"))
        args = snapshot["args"]
        snapshot_dict = snapshot["model"]
        model = resnet18()
        model_dict = model.state_dict()
        snapshot_dict = {k: v for k, v in snapshot_dict.items() if k in model_dict}
        model_dict.update(snapshot_dict)
        model.load_state_dict(model_dict)
        self.model = model
        self.model.eval()
        normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))
        self.transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.roi_extract = Finger_crop(output_w=144, output_h=64, dataset="FV-USM")

    def get_embedding(self, img, is_path=False):
        self.model.eval()
        if is_path:
            img = Image.open(img)
        img = img.transpose(Image.ROTATE_270)
        roi_img, edges = self.roi_extract.crop_finger(np.array(img))
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2RGB)
        roi_img = Image.fromarray(roi_img)
        roi_img = self.transform(roi_img)
        input = torch.unsqueeze(roi_img, 0)
        with torch.no_grad():
            emb = self.model(input)
            emb = preprocessing.normalize(emb.numpy())
            emb = emb.squeeze()
        return emb, edges
