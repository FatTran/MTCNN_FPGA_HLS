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

#define INPUT_SIZE 12
#define PADDING 0
#define CONV1_IN_CHANNEL 3
#define CONV1_SIZE 3
#define CONV1_OUT_SIZE (INPUT_SIZE - CONV1_SIZE + 1)
#define CONV1_FILTER 10
#define MP1_SIZE 3
#define POOL1_OUT_SIZE (CONV1_OUT_SIZE / 2)
#define STRIDE 2

// #define INPUT_SIZE 5
#define PADDING 0
#define CONV2_IN_CHANNEL 10
#define CONV2_SIZE 3
#define CONV2_OUT_SIZE (POOL1_OUT_SIZE - CONV2_SIZE + 1)
#define CONV2_FILTER 16

// #define INPUT_SIZE 3
#define PADDING 0
#define CONV3_IN_CHANNEL 16
#define CONV3_SIZE 3
#define CONV3_OUT_SIZE (CONV2_OUT_SIZE - CONV3_SIZE + 1)
#define CONV3_FILTER 32

// #define INPUT_SIZE 1
#define PADDING 0
#define CONV4_1_IN_CHANNEL 32
#define CONV4_1_SIZE 1
#define CONV4_1_OUT_SIZE (CONV3_OUT_SIZE - CONV4_1_SIZE + 1)
#define CONV4_1_FILTER 2



// #define INPUT_SIZE 1
#define PADDING 0
#define CONV4_2_IN_CHANNEL 32
#define CONV4_2_SIZE 1
#define CONV4_2_OUT_SIZE (CONV3_OUT_SIZE - CONV4_2_SIZE + 1)
#define CONV4_2_FILTER 4
//#define CONV2_SIZE 3
//#define CONV2_FILTER 16
//#define CONV3_SIZE 3
//#define CONV3_FILTER 32
//#define CONV4_1_SIZE 1
//#define CONV4_1_FILTER 2
//#define CONV4_2_SIZE 1
//#define CONV4_2_FILTER 4

//float prelu(float input, float alpha);
//void conv_mp_1_accel(float* input, float* weights, float* output);
//void conv_2_accel(float* input, float* weights, float* output);
//void conv_3_accel(float* input, float* weights, float* output);
//void conv_4_1_accel(float* input, float* weights, float* output);
//void conv_4_2_accel(float* input, float* weights, float* output);
//void pnet_accel(float* input, float* weight1, float* weight2, float* weight3, float* weight4, float* weight5, float* output1, float* output2);

void pnet_accel(float* input, float* output1, float* output2);

