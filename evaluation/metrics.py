import image_similarity_measures
from image_similarity_measures.quality_metrics import psnr, ssim
from imageio import imread
import os
import numpy as np
import torch
from piq import vsi
from torchvision import transforms
from PIL import Image
import cv2


def StainIoU(label_path, image_path):
    light_brown = (0,0,0)
    dark_brown = (50,250,250)

    label = cv2.imread(label_path)                              
    label = cv2.cvtColor(label,cv2.COLOR_BGR2RGB)               
    hsv_label = cv2.cvtColor(label, cv2.COLOR_RGB2HSV)         
    label_mask = cv2.inRange(hsv_label, light_brown, dark_brown)
    label_res = cv2.bitwise_and(label, label, mask=label_mask)  
    label_res = (label_res>0)*255


    output = cv2.imread(image_path)
    output = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
    hsv_output = cv2.cvtColor(output, cv2.COLOR_RGB2HSV)
    output_mask = cv2.inRange(hsv_output, light_brown, dark_brown)
    output_res = cv2.bitwise_and(output, output, mask=output_mask)
    output_res = (output_res>0)*255


    intersection = np.logical_and(label_res, output_res) * 255
    union = np.logical_or(label_res, output_res) * 255

    intersection_count = np.sum(intersection == 255)
    union_count = np.sum(union == 255)

    if intersection_count == 0:
        return 0
    return (intersection_count/union_count)*100

# for vsi
convert_tensor = transforms.ToTensor()

# modify the path here
label_path = r'label path\\'
image_path = r'output path\\'
label_list = os.listdir(label_path)
image_list = os.listdir(image_path)

sum_ssim = 0
sum_psnr = 0
sum_vsi = 0
stain_iou = 0


for i in range(len(image_list)):
    label_dir = label_path+ label_list[i]
    image_dir = image_path+ image_list[i]
    label = imread(label_dir)
    image = imread(image_dir)

    # SSIM and PSNR
    sum_ssim += ssim(label, image)
    sum_psnr += psnr(label, image)
    
    # Stain IoU
    stain_iou += StainIoU(label_dir, image_dir)

    # VSI
    label_tensor = convert_tensor(label).reshape((1, 3, 2048, 2048))
    image_tensor = convert_tensor(image).reshape((1, 3, 2048, 2048))
    vsi_index: torch.Tensor = vsi(label_tensor, image_tensor, data_range=1.)
    sum_vsi += vsi_index

print('SSIM:',round((sum_ssim/len(label_list))*100,2))
print('PSNR:',round(sum_psnr/len(label_list),4))
print('StainIoU:',round(stain_iou/len(label_list),2))
print('VSI:',round((sum_vsi.numpy()/len(label_list))*100,2))