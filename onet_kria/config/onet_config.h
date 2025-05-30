#pragma once
#include "hls_stream.h"
#include "ap_int.h"
#ifndef __SYNTHESIS__
#include <iostream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#endif
#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"
#include "imgproc/xf_channel_extract.hpp"
#include "imgproc/xf_crop.hpp"
#include "imgproc/xf_custom_convolution.hpp"
#include "core/xf_arithm.hpp"
#include "imgproc/xf_channel_combine.hpp"
#include "imgproc/xf_cvt_color.hpp"
#include "imgproc/xf_histogram.hpp"
#include <cmath>
#include "imgproc/xf_convertscaleabs.hpp"
#include "imgproc/xf_lut.hpp"
#include "imgproc/xf_delay.hpp"
#include "imgproc/xf_duplicateimage.hpp"

#define INPUT_SIZE 48
#define PADDING 0
#define CONV1_IN_CHANNEL 3
#define CONV1_SIZE 3
#define CONV1_OUT_SIZE (INPUT_SIZE - CONV1_SIZE + 1)
#define CONV1_FILTER 32
#define MP1_SIZE 3
#define POOL1_OUT_SIZE (CONV1_OUT_SIZE / 2)
#define STRIDE1 2

// #define INPUT_SIZE 5
// #define PADDING 0
#define CONV2_IN_CHANNEL 32
#define CONV2_SIZE 3
#define CONV2_OUT_SIZE (POOL1_OUT_SIZE - CONV2_SIZE + 1)
#define CONV2_FILTER 64
#define MP2_SIZE 3
#define POOL2_OUT_SIZE (CONV2_OUT_SIZE / 2)
#define STRIDE2 2

// #define INPUT_SIZE 3
// #define PADDING 0
#define CONV3_IN_CHANNEL 64
#define CONV3_SIZE 3
#define CONV3_OUT_SIZE (POOL2_OUT_SIZE - CONV3_SIZE + 1)
#define CONV3_FILTER 64
#define MP3_SIZE 2
#define POOL3_OUT_SIZE (CONV3_OUT_SIZE / 2)
#define STRIDE3 2

#define CONV4_IN_CHANNEL 64
#define CONV4_SIZE 2
#define CONV4_OUT_SIZE (POOL3_OUT_SIZE - CONV4_SIZE + 1)
#define CONV4_FILTER 128 


// #define INPUT_SIZE 1
#define FC1_DENSE_SIZE 256

#define FC2_1_DENSE_SIZE 2

#define FC2_2_DENSE_SIZE 4

#define FC2_3_DENSE_SIZE 10

float prelu(float input, float alpha);
void conv_1_accel(float* input, float* output);
// void conv_mp_1_accel(float* input, float* weights, float* output);
// void conv_mp_2_accel(float* input, float* weights, float* output);
// void conv_3_accel(float* input, float* weights, float* output);
// void flatten(float* input, float* output);
// void dense_1_accel(float* input, float* weights, float* output);
// void dense_2_1_accel(float* input, float* weights, float* output);
// void dense_2_2_accel(float* input, float* weights, float* output);
//void pnet_accel(float* input, float* weight1, float* weight2, float* weight3, float* weight4, float* weight5, float* output1, float* output2);

void onet_accel(    float* input, 
                    // float* conv_mp_2_weights, 
                     float* conv_mp_3_weights,
                    // float* conv_4_weights,
                    float* dense_1_weights, 
                    float* output1, 
                    float* output2,
                    float* output3);

