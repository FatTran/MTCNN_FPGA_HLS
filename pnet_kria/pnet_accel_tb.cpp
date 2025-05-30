#include "config/pnet_config.h"
#include "../p_net_weight.h"

void hwc_to_chw(cv::Mat& src, cv::Mat& dst) {
    // Convert HWC to CHW
    std::vector<cv::Mat> channels;
    cv::split(src, channels); // Split the image into its channels

    for (int i = 0; i < src.channels(); ++i) {
    	for (int y = 0; y < channels[i].rows; y++){
    		for(int x = 0; x < channels[i].cols; x++){
    			dst.at<uint8_t>(i, y * channels[i].cols + x) = channels[i].at<uint8_t>(y, x);
    		}
    	}
    }

    // Now chw_img contains the image data in CHW format
    std::cout << "Image converted to CHW format successfully!" << std::endl;
    return;
}

void Mat2Array(cv::Mat& inputMat, float* outputArray) {
    int MatHeight = inputMat.rows;
    int MatWidth = inputMat.cols;
//    std::cout << MatHeight << std::endl;
//    std::cout << MatWidth << std::endl;
//    std::cout << inputMat << std::endl;
    //std::cout << inputMat.at<float>(0, 0);
	for (int y = 0; y < MatHeight; y++) {
		for (int x = 0; x < MatWidth; x++) {
			outputArray[(y) * (MatWidth) + (x)] = inputMat.at<uint8_t>(y, x);
			//std::cout << (int) outputArray[(y) * (MatWidth) + (x)] << std::endl;
		}
	}
	return;
}

int main(int argc, char **argv){
	cv::Mat inputImage = cv::imread(argv[1], cv::IMREAD_COLOR);
	cv::Mat resizedImage;
	cv::Mat RGBImage;
	float input[INPUT_SIZE * INPUT_SIZE * CONV1_IN_CHANNEL] = {0};
	float out1[CONV4_1_OUT_SIZE * CONV4_1_OUT_SIZE * CONV4_1_FILTER] = {0};
	float out2[CONV4_2_OUT_SIZE * CONV4_2_OUT_SIZE * CONV4_2_FILTER] = {0};
	

	cv::resize(inputImage, resizedImage, cv::Size(12,12), 0, 0, cv::INTER_LINEAR);
	cv::Mat CHWImage(resizedImage.channels(), resizedImage.rows * resizedImage.cols, CV_8UC1);
	cv::cvtColor(resizedImage, RGBImage, cv::COLOR_BGR2RGB);
	hwc_to_chw(RGBImage, CHWImage);

	//std::cout << CHWImage << std::endl;
	Mat2Array(CHWImage, &input[0]);



	// conv_mp_1_accel(&input[0], &weight_conv_mp_1[0], &out_pool[0]);
	pnet_accel(	input,
//				weight_conv_mp_1,
//				weight_conv_2,
//				weight_conv_3,
//				weight_conv_4_1,
//				weight_conv_4_2,
				out1,
				out2);


	for(int filter = 0; filter < CONV4_1_FILTER; filter++){
		for(int y = 0 ; y < CONV4_1_OUT_SIZE; y++){
			for(int x = 0; x < CONV4_1_OUT_SIZE; x++){
				std::cout << out1[filter * CONV4_1_OUT_SIZE * CONV4_1_OUT_SIZE + y * CONV4_1_OUT_SIZE + x] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	for(int filter = 0; filter < CONV4_2_FILTER; filter++){
		for(int y = 0 ; y < CONV4_2_OUT_SIZE; y++){
			for(int x = 0; x < CONV4_2_OUT_SIZE; x++){
				std::cout << out2[filter * CONV4_2_OUT_SIZE * CONV4_2_OUT_SIZE + y * CONV4_2_OUT_SIZE + x] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	//max_pool(output_conv, output_pool);

	std::cout << "Finished" << std::endl;
    return 0;
}
