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


#Note: chua co lop softmax trong kernel hardware, co the them softmax tren host

overlay = Overlay("mtcnn_base.bit")

pnet_overlay = overlay.pnet_accel_0
rnet_overlay = overlay.rnet_accel_0
onet_overlay = overlay.onet_accel_0


# print(pnet_overlay.register_map) Kiem tra input, output cua IP
# print(rnet_overlay.register_map) Kiem tra input, output cua IP
# print(onet_overlay.register_map) Kiem tra input, output cua IP

#Lay anh xu ly mot khung hinh pnet
pnet_in = cv2.imread("image.png", cv2.IMREAD_COLOR)

pnet_in = cv2.resize(pnet_in, (12,12), interpolation=cv2.INTER_LINEAR)
pnet_in = util.preprocess(pnet_in)

#Lay anh xu ly mot khung hinh rnet

rnet_in = cv2.imread("image.png", cv2.IMREAD_COLOR)

rnet_in = cv2.resize(rnet_in, (24, 24), interpolation=cv2.INTER_LINEAR) 
rnet_in = util.preprocess(rnet_in)

#Lay anh xu ly mot khung hinh onet

onet_in = cv2.imread("image.png", cv2.IMREAD_COLOR)
onet_in = cv2.resize(onet_in, (48, 48), interpolation=cv2.INTER_LINEAR)
onet_in = util.preprocess(onet_in)

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






