import torch
from pynq import Overlay
from pynq import allocate
import numpy as np
import cv2
import time
from PIL import Image
from utility import util
from matplotlib import pyplot as plt
import rnet_weights
import onet_weights
def pnet_driver(in_image):
    in_image_buffer = allocate(shape=(1,3,12,12), dtype=np.float32, cacheable=False)
    out1_buffer = allocate(shape=(2), dtype=np.float32, cacheable=False)
    out2_buffer = allocate(shape=(4), dtype=np.float32, cacheable=False)
    
    in_image_buffer[:] = in_image
    out1_buffer[:] = np.zeros(shape=(2), dtype=np.float32)
    out2_buffer[:] = np.zeros(shape=(4), dtype=np.float32)
    
    pnet_overlay.register_map.input_r_1 = in_image_buffer.device_address
    pnet_overlay.register_map.output1_1 = out1_buffer.device_address
    pnet_overlay.register_map.output2_1 = out2_buffer.device_address
    HWTimeStart = time.perf_counter()
    if(pnet_overlay.register_map.CTRL.AP_IDLE == 1):
        pnet_overlay.register_map.CTRL.AP_START = 1
    while(pnet_overlay.register_map.CTRL.AP_IDLE != 1):
        continue
    print("HW runtime: ", time.perf_counter() - HWTimeStart)
    out1 = out1_buffer
    out2 = out2_buffer
    
    in_image_buffer.freebuffer()
    out1_buffer.freebuffer()
    out2_buffer.freebuffer()
    return out2, out1

def rnet_driver(in_image,  dense_1_weights):
    in_image_buffer = allocate(shape=(1,3,24,24), dtype=np.float32, cacheable=False)
    
    # conv_mp_2_weights_buffer = allocate(shape=(12096), dtype=np.float32, cacheable=False)
    # conv_3_weights_buffer = allocate(shape=(12288), dtype=np.float32, cacheable=False)
    dense_1_weights_buffer = allocate(shape=(73728), dtype=np.float32, cacheable=False)
    
    out1_buffer = allocate(shape=(2), dtype=np.float32, cacheable=False)
    out2_buffer = allocate(shape=(4), dtype=np.float32, cacheable=False)
    
    in_image_buffer[:] = in_image
    
    # conv_mp_2_weights_buffer[:] = conv_mp_2_weights
    # conv_3_weights_buffer[:] = conv_3_weights
    dense_1_weights_buffer[:] = dense_1_weights
    
    out1_buffer[:] = np.zeros(shape=(2), dtype=np.float32)
    out2_buffer[:] = np.zeros(shape=(4), dtype=np.float32)
    
    rnet_overlay.register_map.input_r_1 = in_image_buffer.device_address
    
    # rnet_overlay.register_map.conv_mp_2_weights_1 = conv_mp_2_weights_buffer.device_address
    # rnet_overlay.register_map.conv_3_weights_1 = conv_3_weights_buffer.device_address
    rnet_overlay.register_map.dense_1_weights_1 = dense_1_weights_buffer.device_address
    
    rnet_overlay.register_map.output1_1 = out1_buffer.device_address
    rnet_overlay.register_map.output2_1 = out2_buffer.device_address
    
    HWTimeStart = time.perf_counter()
    if(rnet_overlay.register_map.CTRL.AP_IDLE == 1):
        rnet_overlay.register_map.CTRL.AP_START = 1
    while(rnet_overlay.register_map.CTRL.AP_IDLE != 1):
        continue
    print("HW runtime: ", time.perf_counter() - HWTimeStart)
    out1 = out1_buffer
    out2 = out2_buffer
    
    in_image_buffer.freebuffer()
    out1_buffer.freebuffer()
    out2_buffer.freebuffer()
    return out2, out1

def onet_driver(in_image, conv_mp_3_weights, dense_1_weights):
    in_image_buffer = allocate(shape=(1,3,48,48), dtype=np.float32, cacheable=False)
    
    # conv_mp_2_weights_buffer = allocate(shape=(18432), dtype=np.float32, cacheable=False)
    conv_mp_3_weights_buffer = allocate(shape=(36864), dtype=np.float32, cacheable=False)
    # conv_4_weights_buffer = allocate(shape=(32768), dtype=np.float32, cacheable=False)
    dense_1_weights_buffer = allocate(shape=(294912), dtype=np.float32, cacheable=False)
    
    out1_buffer = allocate(shape=(2), dtype=np.float32, cacheable=False)
    out2_buffer = allocate(shape=(4), dtype=np.float32, cacheable=False)
    out3_buffer = allocate(shape=(10), dtype=np.float32, cacheable=False)
    
    in_image_buffer[:] = in_image;
    # conv_mp_2_weights_buffer[:] = conv_mp_2_weights
    conv_mp_3_weights_buffer[:] = conv_mp_3_weights
    # conv_4_weights_buffer[:] = conv_4_weights
    dense_1_weights_buffer[:] = dense_1_weights
    
    out1_buffer[:] = np.zeros(shape=(2), dtype=np.float32)
    out2_buffer[:] = np.zeros(shape=(4), dtype=np.float32)
    out3_buffer[:] = np.zeros(shape=(10), dtype=np.float32)
    
    # onet.register_map.input_r_1 = out_conv_buffer.device_address
    
    # onet_overlay.register_map.conv_mp_2_weights_1 = conv_mp_3_weights_buffer.device_address
    onet_overlay.register_map.conv_mp_3_weights_1 = conv_mp_3_weights_buffer.device_address
    # onet_overlay.register_map.conv_4_weights_1 = conv_4_weights_buffer.device_address
    onet_overlay.register_map.dense_1_weights_1 = dense_1_weights_buffer.device_address
    
    onet_overlay.register_map.output1_1 = out1_buffer.device_address
    onet_overlay.register_map.output2_1 = out2_buffer.device_address
    onet_overlay.register_map.output3_1 = out3_buffer.device_address
    
    HWTimeStart = time.perf_counter()
    if(onet_overlay.register_map.CTRL.AP_IDLE == 1):
        onet_overlay.register_map.CTRL.AP_START = 1
    while(onet_overlay.register_map.CTRL.AP_IDLE != 1):
        continue
    print("HW runtime: ", time.perf_counter() - HWTimeStart)
    out1 = out1_buffer
    out2 = out2_buffer
    out3 = out3_buffer
#     print(out1_buffer)
    
    in_image_buffer.freebuffer()
    
    # conv_mp_2_weights_buffer.freebuffer()
    conv_mp_3_weights_buffer.freebuffer()
    # conv_4_weights_buffer.freebuffer()
    dense_1_weights_buffer.freebuffer()
    
    out1_buffer.freebuffer()
    out2_buffer.freebuffer()
    out3_buffer.freebuffer()
    return out3, out2, out1

def crop_image_into_patches(image, patch_size=(12, 12), stride=2):
    # Load the image
    # image = cv2.imread(image_path)
    # if image is None:
        # print("Error: Unable to load image.")
        # return []
    if image is None:
        return []
    patches = []
    img_height, img_width = image.shape[:2]

    # Iterate over the image with the given stride
    for y in range(0, img_height - patch_size[1] + 1, stride):
        for x in range(0, img_width - patch_size[0] + 1, stride):
            patch = image[y:y + patch_size[1], x:x + patch_size[0]]
            patches.append(patch)

    return patches
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)


def pnet_hw(image):
    patch_size = (12,12)
    stride = 2
    batch, channel, img_height, img_width = image.shape
    num_windows_y = (img_height - patch_size[0]) // stride + 1
    num_windows_x = (img_width - patch_size[1]) // stride + 1
    count_x = 0
    count_y = 0

    offset = torch.zeros(1, 4, num_windows_y, num_windows_x)
    prob = torch.zeros(1, 2, num_windows_y, num_windows_x)
    for y in range(0, img_height - patch_size[1] + 1, stride):
        for x in range(0, img_width - patch_size[0] + 1, stride):
            patch = image[y:y + patch_size[1], x:x + patch_size[0]]
            out2, out1 = pnet_driver(patch)
            out1 = softmax(out1)
            for i in range(4):
                offset[0][i][count_y][count_x] = out2[0][i][0][0]
            for j in range(2):
                prob[0][j][count_y][count_x] = out1[0][j][0][0]
            count_x += 1
        count_y += 1
    return offset, prob

def rnet_hw(image_boxes, dense_1_weights):
    patch_size = (24, 24)
    stride = 2
    batch, channel, box_height, box_width  = image_boxes.shape
    offset = torch.zeros(shape=(batch, 4), dtype=np.float32)
    prob = torch.zeros(shape=(batch, 2), dtype=np.float32)
    for i in range(batch):
        out2, out1 = rnet_driver(image_boxes[i][:][:][:], dense_1_weights) 
        out1 = softmax(out1)
        for j in range(4):
            offset[i][j] = out2[j]
        for k in range(2):
            prob[i][k] = out1[k]
    return offset, prob

def onet_hw(image_boxes, conv_mp_3_weights, dense_1_weights):
    patch_size = (48, 48)
    stride = 2
    batch, channel, box_height, box_width = image_boxes.shape
    landmark = torch.zeros(shape=(batch, 10), dtype=np.float32)
    offset = torch.zeros(shape=(batch, 4), dtype=np.float32)
    prob = torch.zeros(shape=(batch, 2), dtype=np.float32)

    for i in range(batch):
        out3, out2, out1 = onet_driver(image_boxes[i][:][:][:], conv_mp_3_weights, dense_1_weights)
        out1 = softmax(out1)
        for j in range(10):
            landmark[i][j] = out3[j]
        for k in range(4):
            offset[i][k] = out2[k]
        for h in range(2):
            prob[i][h] = out1[h] 
    return landmark, offset, prob

# image = np.zero(shape=(720,1280, 3), dtype=np.float32)
# pnet_hw(image)
# overlay = Overlay("mtcnn_base.bit")

# pnet_overlay = overlay.pnet_accel_0
# rnet_overlay = overlay.rnet_accel_0
# onet_overlay = overlay.onet_accel_0


# # print(pnet_overlay.register_map) Kiem tra input, output cua IP
# # print(rnet_overlay.register_map) Kiem tra input, output cua IP
# # print(onet_overlay.register_map) Kiem tra input, output cua IP

# #Lay anh xu ly mot khung hinh pnet
# pnet_in = cv2.imread("image.png", cv2.IMREAD_COLOR)

# pnet_in = cv2.resize(pnet_in, (12,12), interpolation=cv2.INTER_LINEAR)
# pnet_in = util.preprocess(pnet_in)

# #Lay anh xu ly mot khung hinh rnet

# rnet_in = cv2.imread("image.png", cv2.IMREAD_COLOR)

# rnet_in = cv2.resize(rnet_in, (24, 24), interpolation=cv2.INTER_LINEAR) 
# rnet_in = util.preprocess(rnet_in)

# #Lay anh xu ly mot khung hinh onet

# onet_in = cv2.imread("image.png", cv2.IMREAD_COLOR)
# onet_in = cv2.resize(onet_in, (48, 48), interpolation=cv2.INTER_LINEAR)
# onet_in = util.preprocess(onet_in)