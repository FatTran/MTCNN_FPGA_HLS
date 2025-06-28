#include "config/pnet_config.h"
//void printMat(hls::stream<float>& input, int filter, int out_size){
//	for(int i = 0 ; i < filter; i++){
//		for(int j = 0; j < out_size; j++){
//			for(int k = 0 ; k < out_size; k++){
//				std::cout << input.read();
//				std::cout << ",\t";
//			}
//			std::cout << std::endl;
//		}
//		std::cout << std::endl;
//	}
//}
////
//void printMatArr(float* input, int filter, int out_size){
//	for(int i = 0 ; i < filter; i++){
//		for(int j = 0; j < out_size; j++){
//			for(int k = 0 ; k < out_size; k++){
//				std::cout << input[i * out_size * out_size + j * out_size + k];
//				std::cout << ",\t";
//			}
//			std::cout << std::endl;
//		}
//		std::cout << std::endl;
//	}
//}
float relu(float input){
	return (input > 0) ? input : 0;
}
float sigmoid(float input){
	return 1 / (1 + hls::exp(-input));
}

float prelu(float input, float alpha){
    return input > 0 ? input : alpha * input;
}

void macc(float x, float w, float& output){
//#pragma HLS PIPELINE off
	float acc = x * w;
	output = output + acc;
}

void stream2arr(hls::stream<float>& input, float* output, int size){
	for(int i = 0 ; i < size ; i++){
		output[i] = input.read();
	}
	return;
}
void arr2stream(float* inArr, hls::stream<float>& outStream, int size){
	for(int i = 0 ; i < size ; i++){
		outStream.write(inArr[i]);
	}
}
void conv_1_accel(hls::stream<float>& input, hls::stream<float>& output){

	static float bias[8] = {
		-0.208957,
		0.029669,
		-0.223691,
		-0.003911,
		-0.119333,
		0.052083,
		-0.346709,
		0.578722
	};

	static float prelu_weight[8] = {
		0.764446,
		-0.697428,
		0.537747,
		-0.551039,
		0.951688,
		-0.527830,
		-0.203746,
		0.130311
	};
	static float weights[216] = {
		0.713536, 	0.461627, 	0.189490, 
		-0.014683, 	0.466043, 	0.053659, 
		0.036703, 	0.401369, 	0.337439, 

		0.143458, 	-0.086304, 	-0.250425, 
		-0.163726, 	0.233864, 	-0.127791, 
		-0.177009, 	-0.358199, 	-0.243655, 

		0.245394, 	-0.298136, 	0.202663, 
		0.002515, 	0.229467, 	0.333453, 
		-0.032099, 	0.292184, 	-0.018485, 


		0.766388, 	0.288539, 	-0.437410, 
		0.221046, 	-0.099533, 	-0.552111, 
		0.136635, 	-0.193066, 	-0.347425, 

		-0.315995, 	0.207876, 	0.315128, 
		-0.189994, 	0.271459, 	0.489261, 
		-0.451673, 	0.038841, 	0.178962, 

		-0.483281, 	-0.513396, 	0.074273, 
		0.002015, 	-0.103256, 	-0.012000, 
		0.412110, 	0.057647, 	0.216822, 


		0.108048, 	-0.201688, 	-0.284460, 
		0.082070, 	0.599387, 	0.566935, 
		-0.331912, 	-0.412674, 	-0.198741, 

		0.025657, 	-0.021534, 	-0.150161, 
		-0.135075, 	0.596901, 	0.438234, 
		0.046931, 	-0.450942, 	-0.444548, 

		0.027522, 	-0.313567, 	0.003544, 
		0.175720, 	0.368344, 	0.200581, 
		0.043376, 	-0.296959, 	-0.130902, 


		-1.024591, 	-0.241163, 	-0.360946, 
		-0.223878, 	0.216709, 	-0.076845, 
		0.558986, 	0.700525, 	0.409547, 

		0.104286, 	0.325116, 	0.217241, 
		-0.115198, 	0.112781, 	-0.030074, 
		-0.333406, 	-0.016541, 	-0.288999, 

		0.046229, 	0.004589, 	-0.255536, 
		-0.111040, 	0.186885, 	0.041508, 
		-0.070590, 	0.212958, 	-0.051542, 


		0.008830, 	0.210755, 	-0.444955, 
		-0.142656, 	-0.425178, 	-0.131994, 
		0.844683, 	0.453637, 	-0.170731, 

		-0.214225, 	-0.212674, 	0.287069, 
		-0.603671, 	-0.323769, 	0.139908, 
		-0.293294, 	0.169108, 	0.414756, 

		0.382464, 	0.065926, 	-0.088872, 
		-0.223681, 	0.093652, 	-0.059962, 
		0.353604, 	0.320538, 	0.200872, 


		-0.178871, 	-0.524186, 	-0.667984, 
		0.352890, 	0.169089, 	-0.385863, 
		0.233460, 	0.642799, 	0.263255, 

		0.266477, 	-0.013552, 	0.041800, 
		0.138289, 	0.036137, 	-0.327268, 
		0.114743, 	0.130973, 	-0.218678, 

		0.171331, 	-0.056308, 	-0.232826, 
		0.233938, 	-0.052871, 	-0.431418, 
		0.211005, 	0.277497, 	-0.109235, 


		0.305514, 	-0.770592, 	0.144554, 
		0.290771, 	-0.375905, 	0.171569, 
		0.132545, 	0.013559, 	0.250713, 

		0.251808, 	-0.305696, 	0.583983, 
		0.030415, 	-0.047690, 	-0.093363, 
		-0.273728, 	-0.516974, 	-0.371987, 

		0.153078, 	-0.291276, 	0.244273, 
		-0.054521, 	-0.178995, 	0.421589, 
		-0.043532, 	0.103366, 	-0.082871, 


		-0.287275, 	-0.541812, 	-0.270891, 
		-0.258033, 	-0.258226, 	-0.336483, 
		0.137581, 	-0.302634, 	0.064226, 

		0.272086, 	0.224999, 	0.456213, 
		0.144912, 	0.337473, 	0.615987, 
		0.097968, 	0.156749, 	0.285628, 

		0.074957, 	0.290835, 	-0.190830, 
		0.034141, 	0.091532, 	-0.296017, 
		-0.281896, 	0.093066, 	-0.232614
	};
#pragma HLS ARRAY_PARTITION variable=bias type=complete
#pragma HLS ARRAY_PARTITION variable=prelu_weight type=complete

//    float out_conv[CONV1_OUT_SIZE * CONV1_OUT_SIZE * CONV1_FILTER] = {0};
//#pragma HLS BIND_STORAGE variable=out_conv type=ram_s2p impl=lutram
#pragma HLS BIND_STORAGE variable=weights type=ram_s2p impl=lutram
    float sum = 0;
    int in_offset = 0;
    int weight_offset = 0;
    int data_offset = 0;
    float inArr[INPUT_SIZE * INPUT_SIZE * CONV1_IN_CHANNEL] = {0};
    stream2arr(input, inArr, INPUT_SIZE * INPUT_SIZE * CONV1_IN_CHANNEL);

	// Convolution layer
    ConvFilter:
	for (int filter = 0; filter < CONV1_FILTER; filter++) {
		float bias_val = bias[filter];
		float prelu_weight_val = prelu_weight[filter]
;		ConvY:
		for (int ify = 0; ify < CONV1_OUT_SIZE; ify++) {
//#pragma HLS PIPELINE
			ConvX:
			for (int ifx = 0; ifx < CONV1_OUT_SIZE; ifx++){
//#pragma HLS PIPELINE ii=3
				float sumArr[CONV1_SIZE] = {0};
				float sumArr1[CONV1_SIZE] = {0};
				float sumArr2[CONV1_SIZE] = {0};
				float totalSum[CONV1_SIZE] = {0};
				ConvChannel:
				for (int inChan = 0; inChan < CONV1_IN_CHANNEL; inChan++) {
//#pragma HLS PIPELINE
					// ConvKy:
					// for (int ky = 0; ky < CONV1_SIZE; ky++) {
						ConvKx:
						for (int kx = 0; kx < CONV1_SIZE; kx++) {
//#pragma HLS UNROLL
//							float prev = (inChan == 0) ? static_cast<float>(0) : sumArr[kx];
							// sumArr[kx] += inArr[inChan * INPUT_SIZE * INPUT_SIZE + INPUT_SIZE * (ify + ky) + (ifx + kx)] *
									// weights[CONV1_SIZE * CONV1_SIZE * inChan + CONV1_IN_CHANNEL * CONV1_SIZE * CONV1_SIZE * filter + CONV1_SIZE * ky + kx];
							sumArr[kx] += inArr[inChan * INPUT_SIZE * INPUT_SIZE + INPUT_SIZE * (ify + 0) + (ifx + kx)] *
									weights[CONV1_SIZE * CONV1_SIZE * inChan + CONV1_IN_CHANNEL * CONV1_SIZE * CONV1_SIZE * filter + CONV1_SIZE * 0 + kx];
							sumArr1[kx] += inArr[inChan * INPUT_SIZE * INPUT_SIZE + INPUT_SIZE * (ify + 1) + (ifx + kx)] *
									weights[CONV1_SIZE * CONV1_SIZE * inChan + CONV1_IN_CHANNEL * CONV1_SIZE * CONV1_SIZE * filter + CONV1_SIZE * 1 + kx];
							sumArr2[kx] += inArr[inChan * INPUT_SIZE * INPUT_SIZE + INPUT_SIZE * (ify + 2) + (ifx + kx)] *
									weights[CONV1_SIZE * CONV1_SIZE * inChan + CONV1_IN_CHANNEL * CONV1_SIZE * CONV1_SIZE * filter + CONV1_SIZE * 2 + kx];
						}
						for(int i = 0; i < CONV1_SIZE; i++){
							totalSum[i] = sumArr[i]  + sumArr1[i] + sumArr2[i];
						}
					// }
				}
				// output.write(prelu(sumArr[0] + sumArr[1] + sumArr[2] + bias_val, prelu_weight_val));
				output.write(prelu(totalSum[0] + totalSum[1] + totalSum[2] + bias_val, prelu_weight_val));
//				sum = 0;
			}
		}
	}
	return;
}

void mp_1_accel(hls::stream<float>& input, hls::stream<float>& output){
	float out_conv[CONV1_OUT_SIZE * CONV1_OUT_SIZE * CONV1_FILTER] = {0};
#pragma HLS BIND_STORAGE variable=out_conv type=ram_s2p impl=lutram
	int in_offset = 0;
	stream2arr(input, out_conv, CONV1_OUT_SIZE * CONV1_OUT_SIZE * CONV1_FILTER);
//	printMatArr(out_conv, CONV1_FILTER, CONV1_OUT_SIZE);
	//Max pooling layer
	for (int inChan = 0; inChan < CONV1_FILTER; inChan++) {
	Pool_y:
	for (int ify = 0; ify < POOL1_OUT_SIZE; ify++) {
//#pragma HLS PIPELINE off=false ii=35
		Pool_x:
		for (int ifx = 0; ifx < POOL1_OUT_SIZE; ifx++) {
			//float max = FLOAT_MIN;
			Pool_channel:
				int out_conv_offset = inChan * CONV1_OUT_SIZE * CONV1_OUT_SIZE + ify * STRIDE * CONV1_OUT_SIZE + ifx * STRIDE;
				float max = out_conv[out_conv_offset];
				Ky:
				for (int ky = 0; ky < MP1_SIZE; ky++) {
					Kx:
					for (int kx = 0; kx < MP1_SIZE; kx++) {
						in_offset = out_conv_offset + ky * CONV1_OUT_SIZE + kx;
						if( ((CONV1_OUT_SIZE - ifx * STRIDE) >= MP1_SIZE) && ((CONV1_OUT_SIZE - ify * STRIDE) >= MP1_SIZE)){
							max = (out_conv[in_offset] > max) ? out_conv[in_offset] : max;
						} else {
							if(kx >= (CONV1_OUT_SIZE - ifx * STRIDE)){
								in_offset = in_offset - 1;
							}
							if(ky >= (CONV1_OUT_SIZE - ify * STRIDE)){
								in_offset = in_offset - CONV1_OUT_SIZE;
							}
							max = (out_conv[in_offset] > max) ? out_conv[in_offset] : max;
						}
					}
				}
				output.write(max);
			}
		}
	}

	return;
}

void conv_2_accel(hls::stream<float>& input, hls::stream<float>& output){
	static float prelu_weight[16] = {
			0.109032,
			0.380678,
			0.415567,
			-1.467588,
			0.701968,
			0.875446,
			-0.728521,
			1.115470,
			0.735095,
			-0.048331,
			-0.587577,
			0.520378,
			0.348214,
			0.644997,
			0.757137,
			1.176689
	};

	float dw_weight[72] = {
			0.206287, 	0.489257, 	0.227083,
			-0.041010, 	0.103347, 	-0.015091,
			-0.180226, 	-0.416605, 	-0.364940,


			0.090483, 	0.106485, 	0.219278,
			0.285454, 	0.264332, 	0.501651,
			0.093764, 	0.428941, 	0.347074,


			0.070614, 	0.005579, 	0.047983,
			0.207477, 	0.368070, 	0.297661,
			0.158694, 	0.433051, 	0.302856,


			0.546674, 	0.402696, 	0.420204,
			0.060636, 	0.141725, 	0.121510,
			-0.249013, 	-0.349288, 	-0.190513,


			-0.298563, 	0.054891, 	0.068455,
			-0.541069, 	0.166481, 	0.464679,
			-0.384087, 	0.148871, 	0.344093,


			0.075818, 	0.134336, 	-0.630948,
			-0.250213, 	0.050353, 	-0.132405,
			-0.390495, 	0.194372, 	0.236522,


			-0.490623, 	-0.480754, 	-0.494097,
			-0.080591, 	0.010558, 	-0.078319,
			-0.268326, 	-0.055211, 	-0.202027,


			0.189148, 	-0.051829, 	-0.378457,
			-0.114551, 	-0.345460, 	-0.191673,
			0.385747, 	-0.393290, 	-0.158586
	};
	float pw_weight[128] = {
		0.301532, 
		1.135435, 
		0.247129, 
		-0.409294, 
		-0.068401, 
		0.043662, 
		-0.541902, 
		-0.826223, 
		-0.052804, 
		0.075013, 
		-0.216531, 
		0.528948, 
		-1.119988, 
		0.463444, 
		-0.005466, 
		-0.240936, 
		0.045856, 
		0.561706, 
		-0.887348, 
		0.196158, 
		-0.175679, 
		0.336187, 
		0.112969, 
		1.197264, 
		-0.357726, 
		0.501434, 
		0.113869, 
		-0.150031, 
		-0.134097, 
		0.322904, 
		-0.749862, 
		-0.147167, 
		-0.236763, 
		-0.793586, 
		0.409372, 
		0.177783, 
		0.056234, 
		-0.114548, 
		-0.512619, 
		1.186902, 
		-0.409217, 
		0.101566, 
		-0.175406, 
		-0.525432, 
		-0.234428, 
		0.892304, 
		-0.553190, 
		-0.262780, 
		0.889645, 
		-0.068455, 
		-0.280807, 
		-0.047694, 
		0.006333, 
		-0.192112, 
		-0.225063, 
		-0.070910, 
		-0.614306, 
		-0.531500, 
		0.440753, 
		0.467083, 
		0.021457, 
		0.249899, 
		-0.183657, 
		0.537196, 
		0.075510, 
		0.378716, 
		-0.424926, 
		0.827796, 
		-0.392199, 
		-0.215420, 
		0.342696, 
		-0.118663, 
		0.064527, 
		-0.226955, 
		-1.315189, 
		0.144312, 
		0.111006, 
		-0.140103, 
		-0.050537, 
		0.039280, 
		-0.127333, 
		0.114627, 
		0.017835, 
		0.295489, 
		1.284187, 
		0.138987, 
		-0.024700, 
		-0.075534, 
		-0.933106, 
		0.565864, 
		-0.118627, 
		0.025217, 
		0.407228, 
		-0.100489, 
		0.209439, 
		-0.313812, 
		-0.491504, 
		0.297959, 
		-0.409019, 
		-0.654169, 
		0.441290, 
		-0.311054, 
		-0.586054, 
		0.100308, 
		0.237189, 
		-0.133941, 
		0.703050, 
		-0.417877, 
		-0.250029, 
		0.120871, 
		0.754224, 
		-0.552181, 
		0.649217, 
		0.136209, 
		0.140967, 
		-1.078284, 
		0.841842, 
		-0.214424, 
		-0.167028, 
		0.053665, 
		-0.042060, 
		-0.014796, 
		0.463300, 
		0.805068, 
		-0.587996, 
		0.356785, 
		-0.024227, 
		0.347653,
	};
//#pragma HLS ARRAY_PARTITION variable=bias type=complete
#pragma HLS ARRAY_PARTITION variable=prelu_weight type=complete

	float depthOut[CONV2_OUT_SIZE * CONV2_OUT_SIZE * CONV2_IN_CHANNEL] = {0};
	float inArr[POOL1_OUT_SIZE * POOL1_OUT_SIZE * CONV1_FILTER] = {0};
	stream2arr(input, inArr, POOL1_OUT_SIZE * POOL1_OUT_SIZE * CONV1_FILTER);

	//Depthwise convolution
	ConvFilter:
	for (int filter = 0; filter < CONV2_IN_CHANNEL; filter++) {
		ConvY:
		for (int ify = 0; ify < CONV2_OUT_SIZE; ify++) {
#pragma HLS PIPELINE ii=120
			ConvX:
			for (int ifx = 0; ifx < CONV2_OUT_SIZE; ifx++){
				int data_offset = CONV2_OUT_SIZE * CONV2_OUT_SIZE * filter + ify * CONV2_OUT_SIZE  + ifx;
				float sumArr[CONV2_SIZE] = {0};
				ConvKy:
				for (int ky = 0; ky < CONV2_SIZE; ky++) {
					ConvKx:
					for (int kx = 0; kx < CONV2_SIZE; kx++) {
						sumArr[kx] += inArr[filter * POOL1_OUT_SIZE * POOL1_OUT_SIZE + POOL1_OUT_SIZE * (ify + ky) + (ifx + kx)] *
							dw_weight[CONV2_SIZE * CONV2_SIZE * filter + CONV2_SIZE * ky + kx];
					}
				}
				depthOut[data_offset] = sumArr[0] + sumArr[1] + sumArr[2];
			}
		}
	}

	//Pointwise convolution
	ConvFilterPW:
	for(int filter = 0 ; filter < CONV2_FILTER; filter++){
		float prelu_weight_val = prelu_weight[filter];
		ConvYPW:
		for(int ify = 0 ; ify < CONV2_OUT_SIZE; ify++){
			ConvXPW:
			for(int ifx = 0; ifx < CONV2_OUT_SIZE; ifx++){
				float sum = 0;
				int data_offset = CONV2_OUT_SIZE * CONV2_OUT_SIZE * filter + ify * CONV2_OUT_SIZE + ifx;
				ConvChanPW:
				for(int inChan = 0; inChan < CONV2_IN_CHANNEL; inChan++){
					sum += depthOut[inChan * CONV2_OUT_SIZE * CONV2_OUT_SIZE + CONV2_OUT_SIZE * ify + ifx] *
				pw_weight[CONV2_IN_CHANNEL * filter + inChan];
				}
				output.write(prelu(sum, prelu_weight_val));
			}
		}
	}
	return;
}

void conv_3_accel(hls::stream<float>& input, hls::stream<float>& output){
//#pragma HLS ARRAY_PARTITION variable=bias type=complete
//#pragma HLS ARRAY_PARTITION variable=prelu_weight type=complete

	float dw_weight[144] = {
		-0.493691, 	0.148136, 	0.126778, 
		-0.490954, 	-0.012431, 	-0.275314, 
		-0.314460, 	0.074284, 	0.534915, 


		-0.384279, 	-0.293513, 	-0.056185, 
		-0.144119, 	-0.004139, 	0.286188, 
		-0.456364, 	0.022688, 	0.197559, 


		-0.054384, 	-0.385466, 	-0.874687, 
		0.173588, 	0.006501, 	0.077821, 
		-0.057150, 	-0.130631, 	0.040148, 


		0.125023, 	0.330450, 	0.191692, 
		0.099546, 	0.279911, 	0.368376, 
		0.230163, 	0.429018, 	0.374749, 


		-0.007995, 	-0.044817, 	-0.303196, 
		-0.072823, 	0.300127, 	-0.104559, 
		0.443510, 	0.836201, 	-0.307858, 


		0.374119, 	0.568080, 	0.139703, 
		0.098749, 	0.160509, 	-0.138547, 
		0.317308, 	0.609396, 	0.257263, 


		0.400806, 	0.247233, 	0.394962, 
		0.422416, 	0.131008, 	0.345167, 
		0.235345, 	0.051613, 	-0.095014, 


		-0.309279, 	-0.402125, 	-0.109706, 
		0.018432, 	-0.243802, 	-0.085783, 
		0.276003, 	-0.639288, 	0.257481, 


		-0.251516, 	-0.188571, 	-0.543046, 
		-0.197097, 	-0.216603, 	-0.299264, 
		-0.132613, 	0.107223, 	-0.083407, 


		-0.486450, 	-0.493164, 	-0.316324, 
		-0.422726, 	-0.046818, 	-0.141588, 
		-0.848963, 	-0.232865, 	-0.768593, 


		0.089464, 	0.363468, 	0.258568, 
		0.059407, 	-0.065590, 	-0.040354, 
		-0.402222, 	-0.488094, 	-0.384400, 


		-0.198793, 	0.186260, 	-0.251743, 
		-0.434296, 	0.325384, 	-0.164865, 
		0.166743, 	0.452068, 	0.701561, 


		0.385751, 	0.477518, 	-0.101306, 
		0.147653, 	0.237140, 	-0.165955, 
		0.502331, 	0.374589, 	0.152342, 


		0.412463, 	-0.277198, 	0.291305, 
		0.173625, 	-0.338005, 	0.288516, 
		0.403567, 	-0.113660, 	0.338662, 


		-0.358242, 	-0.122541, 	-0.414119, 
		0.183970, 	0.330210, 	0.607257, 
		-0.011643, 	-0.048993, 	0.235801, 


		0.145686, 	-0.266775, 	-0.755204, 
		0.207284, 	0.176711, 	0.130494, 
		-0.229296, 	-0.023134, 	-0.199346
	};
	float pw_weight[512] = {
		-0.235861, 
		-0.057153, 
		-0.441066, 
		-0.279498, 
		-0.666955, 
		-0.060422, 
		-0.036169, 
		-0.598813, 
		-0.084593, 
		-0.370129, 
		0.462457, 
		0.773652, 
		-0.252205, 
		0.501090, 
		-0.657612, 
		-0.153024, 
		-0.274342, 
		0.230107, 
		-0.351841, 
		0.338376, 
		-0.739224, 
		0.486874, 
		-0.083123, 
		0.787114, 
		0.283567, 
		0.113431, 
		0.074768, 
		-0.105243, 
		-0.374566, 
		0.364181, 
		-0.048885, 
		-0.451128, 
		0.096951, 
		0.585894, 
		0.051956, 
		0.116335, 
		-0.211183, 
		-0.070475, 
		-0.476830, 
		-0.094706, 
		-0.672076, 
		0.149353, 
		-0.109944, 
		-0.139022, 
		-0.196056, 
		0.114019, 
		0.758949, 
		0.096486, 
		1.035298, 
		0.053230, 
		0.256597, 
		0.093083, 
		-0.572599, 
		0.073521, 
		-0.032651, 
		-0.201812, 
		0.064604, 
		-0.042820, 
		-0.201447, 
		0.232421, 
		0.095676, 
		0.309792, 
		0.156909, 
		0.221231, 
		0.351103, 
		0.019583, 
		0.613703, 
		0.022544, 
		0.725353, 
		-0.306286, 
		-0.172991, 
		-0.017782, 
		-0.384402, 
		0.405614, 
		-0.053684, 
		-0.059389, 
		-0.162594, 
		-0.658458, 
		0.205480, 
		0.006053, 
		0.196366, 
		-0.127489, 
		0.741891, 
		-0.110183, 
		0.404284, 
		-0.266709, 
		-0.450354, 
		0.033526, 
		-0.683109, 
		0.099458, 
		-0.303480, 
		-0.217975, 
		-0.123561, 
		-0.329068, 
		-0.269476, 
		0.372229, 
		-0.084545, 
		-0.142549, 
		-0.488644, 
		0.409402, 
		0.022446, 
		0.125841, 
		-0.013176, 
		-0.158519, 
		0.343270, 
		-0.380501, 
		-0.220382, 
		0.317276, 
		0.015203, 
		0.319621, 
		0.149588, 
		0.391820, 
		0.103319, 
		-0.126481, 
		0.025336, 
		0.492723, 
		-0.404583, 
		0.272746, 
		0.089387, 
		0.416094, 
		0.151831, 
		-0.273571, 
		0.334604, 
		-0.566749, 
		0.256215, 
		0.086384, 
		0.223037, 
		-0.235807, 
		0.146899, 
		-0.053002, 
		0.060264, 
		-0.177724, 
		0.155731, 
		0.170064, 
		0.040792, 
		0.575054, 
		0.183092, 
		0.130267, 
		0.234427, 
		0.643518, 
		-0.862826, 
		-0.216832, 
		-0.372758, 
		-0.109344, 
		-0.065190, 
		-0.061800, 
		-0.568518, 
		-0.036140, 
		0.268781, 
		0.050802, 
		0.729490, 
		0.278224, 
		0.470778, 
		-0.673771, 
		0.409437, 
		-0.156502, 
		-0.095568, 
		0.381752, 
		0.679946, 
		0.633733, 
		-0.314914, 
		0.036125, 
		-0.156943, 
		-0.323039, 
		-0.576697, 
		0.081938, 
		-0.037024, 
		-0.241317, 
		0.359573, 
		-0.430140, 
		-0.121306, 
		0.362953, 
		-0.074620, 
		-0.509712, 
		-0.124897, 
		0.171594, 
		-0.591271, 
		0.221206, 
		-0.095168, 
		-0.428175, 
		0.396222, 
		0.206748, 
		0.465354, 
		-0.338095, 
		0.113183, 
		0.209248, 
		0.234613, 
		0.620816, 
		0.188413, 
		-0.035289, 
		0.392287, 
		-0.167473, 
		0.230559, 
		-0.069315, 
		0.177548, 
		0.217782, 
		0.227609, 
		0.412341, 
		0.175499, 
		-0.370805, 
		-0.173622, 
		-0.114467, 
		0.568531, 
		0.302531, 
		0.135344, 
		0.290343, 
		-0.272486, 
		-0.242245, 
		-0.460908, 
		-0.724777, 
		-0.447885, 
		0.588729, 
		-0.281764, 
		0.213619, 
		0.168703, 
		0.241903, 
		0.103464, 
		0.127609, 
		0.210034, 
		-0.134877, 
		0.086273, 
		0.293995, 
		0.240737, 
		0.196253, 
		-0.197729, 
		-0.146185, 
		0.251188, 
		0.143435, 
		-0.638346, 
		0.700692, 
		0.276595, 
		0.593006, 
		0.212274, 
		0.020809, 
		-0.020003, 
		-0.206463, 
		-0.212763, 
		0.309495, 
		-0.112865, 
		-0.464158, 
		0.982616, 
		0.198624, 
		-0.597261, 
		0.002780, 
		-0.546832, 
		0.078040, 
		0.437307, 
		-0.457273, 
		0.436753, 
		0.217186, 
		0.357106, 
		-0.080774, 
		0.416345, 
		-0.107828, 
		-0.242394, 
		-0.106770, 
		0.356579, 
		-0.002439, 
		0.205132, 
		0.203083, 
		-0.162346, 
		0.202203, 
		0.335665, 
		0.165530, 
		-0.141154, 
		0.376750, 
		0.489784, 
		-0.012905, 
		0.430096, 
		0.272779, 
		0.169968, 
		0.232184, 
		0.571195, 
		-0.062482, 
		0.826775, 
		-0.124207, 
		0.239230, 
		-0.401743, 
		-0.619850, 
		0.328084, 
		-0.346743, 
		0.246289, 
		-0.532333, 
		0.057360, 
		-0.203804, 
		-0.591041, 
		-0.687643, 
		0.959523, 
		-0.124140, 
		-0.605369, 
		0.698964, 
		-0.386182, 
		0.492800, 
		-0.117470, 
		-0.133273, 
		-0.100473, 
		-0.490022, 
		-0.171873, 
		0.168732, 
		-0.540277, 
		0.085204, 
		-0.158891, 
		0.014928, 
		0.162587, 
		0.285242, 
		-0.298153, 
		-0.393543, 
		0.247876, 
		-0.473992, 
		0.337528, 
		0.067496, 
		-0.300415, 
		-0.325804, 
		0.057471, 
		-0.032944, 
		0.252516, 
		0.175089, 
		0.464323, 
		-0.422672, 
		-0.755444, 
		-0.315032, 
		0.325470, 
		-0.528288, 
		-0.158649, 
		-0.196522, 
		-0.079885, 
		-0.601952, 
		0.015148, 
		-0.584785, 
		-0.020075, 
		-0.143977, 
		-0.325015, 
		-0.251928, 
		0.138609, 
		0.034268, 
		0.603537, 
		0.247992, 
		-0.050345, 
		-0.172999, 
		-0.333944, 
		-0.193063, 
		0.294317, 
		0.155301, 
		-0.003189, 
		-0.196282, 
		-0.785655, 
		0.347952, 
		-0.182717, 
		-0.471159, 
		0.785571, 
		-0.299659, 
		-0.313599, 
		-0.055439, 
		-0.333725, 
		-0.226002, 
		-0.052437, 
		-1.134589, 
		0.143907, 
		0.212983, 
		-0.310424, 
		0.121987, 
		0.181899, 
		0.954130, 
		-0.181642, 
		-0.028089, 
		-0.186611, 
		-0.260298, 
		-0.319604, 
		-0.432381, 
		-0.250656, 
		0.028546, 
		-0.415014, 
		0.267634, 
		0.007795, 
		-0.469765, 
		-0.320979, 
		-0.046267, 
		0.234000, 
		-0.170605, 
		-0.461861, 
		0.413986, 
		-0.008669, 
		-0.162491, 
		-0.149492, 
		-0.583302, 
		-0.444946, 
		-0.413067, 
		0.046397, 
		-0.183394, 
		0.213603, 
		0.089172, 
		-0.626304, 
		-0.081166, 
		-0.524028, 
		0.277992, 
		0.208077, 
		0.221397, 
		1.070051, 
		-0.217163, 
		-0.012053, 
		-0.499018, 
		-0.116862, 
		-0.313881, 
		0.471741, 
		-0.682254, 
		0.075442, 
		0.167569, 
		0.251139, 
		-0.033704, 
		-0.906099, 
		0.066549, 
		0.171663, 
		0.435966, 
		0.760058, 
		-0.080587, 
		-0.121624, 
		-0.083438, 
		0.082857, 
		0.167758, 
		0.026569, 
		-0.167827, 
		0.318799, 
		-0.267200, 
		0.173888, 
		0.314208, 
		-0.256403, 
		0.055995, 
		-0.737252, 
		0.201578, 
		-0.590037, 
		-0.209136, 
		-0.495351, 
		-0.711265, 
		-0.065584, 
		-0.532744, 
		0.046673, 
		-0.595408, 
		0.046245, 
		1.102666, 
		-1.094825, 
		0.237638, 
		-0.571051, 
		0.309863, 
		-0.487579, 
		0.151442, 
		0.519052, 
		0.653025, 
		-0.984039, 
		0.630060, 
		0.391447, 
		0.099919, 
		0.486315, 
		-0.295189, 
		-0.069039, 
		-0.172090, 
		0.277676, 
		0.057654, 
		-0.230175, 
		0.026314, 
		0.265127, 
		-0.484870, 
		-0.140399, 
		0.264225, 
		0.227461, 
		-0.510512, 
		-0.341429, 
		0.199236, 
		0.354556, 
		-0.220028, 
		0.468185, 
		-0.102050, 
		0.062754, 
		0.024299, 
		-0.346383, 
		0.028313, 
		-0.350373, 
		-0.106678, 
		0.263501, 
		-0.019035, 
		0.010051, 
		0.234792, 
		-0.376738, 
		-0.181143, 
		-0.261503, 
		0.282165, 
		0.427992, 
		0.372609, 
		0.607254, 
		-0.213750, 
		-0.299976, 
		0.386655, 
		-0.273503, 
		0.448434, 
		-0.009204, 
		0.539103, 
		0.318066, 
		-0.281702, 
		-0.515481, 
		-0.873818, 
		-0.259709, 
		0.090064, 
		-0.197970, 
		0.029441, 
		0.548283, 
		0.236243, 
		-0.182101, 
		0.120620, 
		-0.936567, 
		0.077124, 
		0.443670, 
		0.591958, 
		0.340389
	};
	float prelu_weight[32] = {
		-0.134541,
		0.067442,
		-0.196624,
		0.013391,
		1.154613,
		0.950656,
		0.008050,
		0.057683,
		0.055582,
		0.177050,
		-0.025046,
		-0.268167,
		-0.025300,
		-0.046192,
		0.150217,
		0.456938,
		-0.093176,
		0.173956,
		0.512766,
		0.061375,
		0.271042,
		0.049593,
		0.049616,
		0.219557,
		-0.052020,
		0.829200,
		0.075918,
		0.832271,
		-0.424625,
		0.000177,
		0.045445,
		0.020690
	};
	float depth_out[CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_IN_CHANNEL] = {0};
	float inArr[CONV2_OUT_SIZE * CONV2_OUT_SIZE * CONV2_FILTER] = {0};
	stream2arr(input, inArr, CONV2_OUT_SIZE * CONV2_OUT_SIZE * CONV2_FILTER);
	//Depthwise convolution
	for(int filter = 0 ; filter < CONV3_IN_CHANNEL; filter++){
#pragma HLS PIPELINE ii=9
		for(int ify = 0; ify < CONV3_OUT_SIZE; ify++){
			for(int ifx = 0; ifx < CONV3_OUT_SIZE; ifx++){
				int data_offset = CONV3_OUT_SIZE * CONV3_OUT_SIZE * filter + CONV3_OUT_SIZE * ify + ifx;
				float sumArr[CONV3_SIZE] = {0};
				for(int ky = 0; ky < CONV3_SIZE; ky++){
					for(int kx = 0 ; kx < CONV3_SIZE; kx++){
						sumArr[kx] += inArr[filter * CONV2_OUT_SIZE * CONV2_OUT_SIZE + CONV2_OUT_SIZE * (ify + ky) + (ifx + kx)] * dw_weight[CONV3_SIZE * CONV3_SIZE * filter + CONV3_SIZE * ky + kx];
					}
				}
				depth_out[data_offset] = sumArr[0] + sumArr[1] + sumArr[2];
			}
		}
	}

	//Pointwise convolution
	ConvFilter:
	for(int filter = 0 ; filter < CONV3_FILTER; filter++){
#pragma HLS PIPELINE ii=83
		ConvY:
		for(int ify = 0 ; ify < CONV3_OUT_SIZE; ify++){
			ConvX:
			for(int ifx = 0 ; ifx < CONV3_OUT_SIZE; ifx++){
				int data_offset = filter * CONV3_OUT_SIZE * CONV3_OUT_SIZE + CONV3_OUT_SIZE * ify + ifx;
				float sum = 0;
				ConvChan:
				for(int inChan = 0; inChan < CONV3_IN_CHANNEL; inChan++){
					sum += depth_out[inChan * CONV3_OUT_SIZE * CONV3_OUT_SIZE + CONV3_OUT_SIZE * ify + ifx] * pw_weight[filter * CONV3_IN_CHANNEL + inChan];
				}
				output.write(prelu(sum, prelu_weight[filter]));
			}
		}
	}
	return;
}

void SEBlock(hls::stream<float>& input, hls::stream<float>& output){
	float inArr[CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_FILTER] = {0};
	stream2arr(input, inArr, CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_FILTER);
	//Fully connected 1
	float seb_fc1_out[8] = {0};
	float seb_fc2_out[32] = {0};
	float seb_fc1_weight[256] = {
		-0.574134, 	-0.563155, 	0.250058, 	0.961377, 	0.584061, 	-0.323477, 	0.383562, 	-0.013974, 	1.060778, 	0.926115, 	-0.373923, 	0.708492, 	-0.213433, 	-0.641659, 	-0.653087, 	0.674744, 	0.434102, 	0.268511, 	-0.221838, 	0.058715, 	-0.759935, 	-0.498462, 	0.085377, 	-0.422129, 	-0.612193, 	-0.611386, 	-0.826781, 	0.189923, 	0.590517, 	-0.663625, 	0.141896, 	-0.120904, 
		0.000645, 	0.374687, 	-0.026352, 	-0.664524, 	0.478954, 	-0.051305, 	-0.188880, 	-0.004974, 	-0.328944, 	0.210398, 	0.059784, 	-1.145202, 	-0.040715, 	0.415492, 	-0.281164, 	-0.466354, 	-0.090302, 	-0.100790, 	0.030562, 	-0.142883, 	-0.193731, 	-0.229150, 	-0.221016, 	0.218191, 	0.582391, 	0.791436, 	0.348389, 	-0.048885, 	0.278601, 	-0.299225, 	0.788495, 	0.394751, 
		0.091498, 	0.114509, 	0.295941, 	-0.503702, 	0.471635, 	0.270666, 	-0.139273, 	-0.106850, 	-0.132316, 	0.457891, 	0.683336, 	-0.723372, 	0.330441, 	-0.342378, 	0.592893, 	-0.582538, 	-0.393955, 	0.343907, 	0.304595, 	-0.498276, 	-0.261126, 	-0.093119, 	0.106498, 	0.517773, 	-0.473098, 	-0.565746, 	0.907753, 	0.049414, 	0.252221, 	-0.172332, 	0.769637, 	0.195580, 
		0.314128, 	0.001266, 	0.648248, 	-0.588774, 	0.196653, 	-0.245286, 	0.547450, 	0.061108, 	0.707956, 	-0.390738, 	-0.196683, 	-0.267468, 	0.219931, 	-0.367301, 	0.326729, 	0.602149, 	-0.102936, 	0.463957, 	0.311209, 	-0.019035, 	-0.329301, 	0.783481, 	0.124788, 	0.747244, 	-0.338767, 	-0.150599, 	0.629854, 	-0.045076, 	-0.456316, 	0.097417, 	0.676246, 	0.517071, 
		0.563252, 	0.023864, 	0.609934, 	0.180625, 	0.128201, 	0.091945, 	-0.321637, 	0.360842, 	0.147200, 	0.185869, 	-0.199779, 	0.949937, 	0.033929, 	0.255463, 	-0.209188, 	-0.073326, 	0.890343, 	0.925654, 	-0.423626, 	0.003709, 	0.043940, 	-0.598768, 	0.534700, 	-0.271319, 	0.157726, 	0.512820, 	-0.324698, 	-0.076941, 	0.424326, 	0.257233, 	-0.205730, 	0.179594, 
		0.867942, 	-0.840450, 	0.137802, 	-0.912473, 	0.230033, 	0.107642, 	0.117646, 	0.378577, 	-1.292602, 	0.344113, 	0.599960, 	0.480566, 	0.738846, 	0.839191, 	-0.522576, 	-0.571132, 	0.630144, 	0.038240, 	0.467397, 	0.580077, 	-1.047001, 	-1.263529, 	0.401169, 	-0.472954, 	-0.390887, 	0.734328, 	0.198707, 	0.397509, 	-0.805956, 	-0.323845, 	-0.124143, 	-0.102334, 
		0.018208, 	0.234320, 	-0.231624, 	0.496969, 	-0.303409, 	-0.078517, 	0.109516, 	0.463400, 	-0.762811, 	-0.903010, 	-0.200879, 	0.128976, 	0.435300, 	-0.351826, 	0.233379, 	0.368522, 	0.106460, 	-0.122764, 	-0.190564, 	-0.008668, 	0.505213, 	-0.811812, 	0.713947, 	0.629374, 	0.343512, 	0.349989, 	-0.680748, 	0.423820, 	0.739987, 	-0.065278, 	-0.814252, 	-0.251985, 
		0.955510, 	-0.548742, 	-0.121329, 	0.469530, 	0.672609, 	0.203131, 	-0.141044, 	0.558024, 	-0.299845, 	-0.193021, 	0.259079, 	-0.266998, 	0.649661, 	-0.035823, 	-0.512420, 	0.645297, 	0.010647, 	0.104860, 	0.460762, 	0.917425, 	-1.227626, 	0.327389, 	0.635646, 	-0.350038, 	-0.255713, 	0.229609, 	-0.448857, 	-0.042315, 	-0.100845, 	-0.858572, 	-0.028461, 	-0.343539
	};
	float seb_fc2_weight[256] = {
		-0.148226, 	-0.056227, 	-0.436285, 	0.261262, 	-0.636387, 	-0.110712, 	-0.483253, 	-0.035767, 
		-0.331571, 	0.064559, 	-0.124614, 	0.197980, 	-0.951199, 	-0.147970, 	0.508849, 	-0.041646, 
		0.070660, 	-0.275101, 	0.053552, 	-0.874240, 	-0.137309, 	-0.387352, 	0.099539, 	-0.334527, 
		-0.334899, 	-0.402392, 	-0.699149, 	0.003734, 	-0.134041, 	-0.138894, 	-0.596637, 	-0.388489, 
		0.261780, 	-0.870704, 	0.088097, 	0.112053, 	-0.226984, 	-0.447088, 	0.348585, 	-0.175056, 
		0.262523, 	-0.200429, 	-0.026460, 	-0.009686, 	-0.470367, 	-0.776683, 	0.580869, 	-0.341091, 
		-0.041035, 	0.131929, 	0.363636, 	-0.913756, 	-0.596392, 	0.055570, 	-0.345516, 	-0.117811, 
		-0.552513, 	0.198460, 	0.588028, 	0.790569, 	-0.638706, 	0.268752, 	-0.252343, 	-0.560605, 
		-0.005076, 	0.213842, 	-0.732974, 	-1.149338, 	-0.364136, 	0.318540, 	-0.830181, 	-0.065027, 
		-0.611225, 	0.132971, 	0.673417, 	0.118133, 	0.277309, 	-0.816651, 	-0.072500, 	-0.326029, 
		-1.048536, 	0.104386, 	-0.769347, 	0.511808, 	0.103908, 	0.111041, 	0.105818, 	-0.101258, 
		-0.900842, 	0.050316, 	-0.003819, 	-0.090635, 	-0.702257, 	-0.200079, 	0.053684, 	-1.767355, 
		-0.111047, 	0.190729, 	-1.042202, 	-0.085592, 	0.229290, 	-0.580087, 	-0.584578, 	-0.424478, 
		-0.256903, 	-0.381269, 	0.301759, 	-0.018340, 	-0.588485, 	-0.292743, 	-0.178442, 	-0.611567, 
		-1.017737, 	0.009485, 	0.446841, 	0.084804, 	-0.596462, 	-0.247235, 	0.174002, 	-0.313169, 
		-0.907528, 	0.036578, 	-0.530142, 	-0.041287, 	-0.602091, 	-0.357142, 	-1.703814, 	-0.605522, 
		-0.710858, 	0.251410, 	0.142689, 	0.715539, 	-0.338466, 	0.000279, 	0.123007, 	-0.426970, 
		-0.285210, 	0.243787, 	0.116144, 	0.033345, 	-0.546619, 	-0.867331, 	-0.269999, 	-0.238807, 
		0.248680, 	0.275413, 	-0.091650, 	-0.395661, 	-0.303628, 	-0.495448, 	-0.279098, 	-0.333771, 
		-1.381924, 	-0.150965, 	-0.589397, 	0.176707, 	-0.043801, 	-0.676060, 	-0.682801, 	-0.322924, 
		-0.833830, 	0.073098, 	-1.060719, 	-0.489134, 	-0.046934, 	-0.485232, 	-0.021744, 	-1.002244, 
		-0.273727, 	-0.443591, 	0.220771, 	0.397700, 	-0.456960, 	-0.371742, 	0.254997, 	0.501222, 
		-0.443917, 	-0.787560, 	-0.054834, 	0.624285, 	0.268149, 	0.153294, 	-0.706675, 	-0.183025, 
		-0.718434, 	-0.144072, 	-0.810983, 	-0.487809, 	-0.135668, 	0.106523, 	0.305272, 	-0.610152, 
		0.809028, 	-0.790066, 	-0.375372, 	0.159349, 	-0.161590, 	-0.441196, 	0.043322, 	-0.209812, 
		-0.152144, 	0.186805, 	0.206069, 	0.046295, 	-0.487657, 	0.478664, 	-0.375742, 	0.430107, 
		0.151741, 	-1.225752, 	-0.733168, 	-0.089549, 	0.002918, 	-0.735392, 	0.294125, 	-0.100880, 
		-0.544208, 	-0.318261, 	0.207948, 	0.167401, 	-0.909030, 	-1.093186, 	-0.669623, 	-0.548963, 
		-0.331069, 	-0.122786, 	-0.548226, 	-0.038565, 	-0.471983, 	-0.077246, 	-0.255032, 	-0.298493, 
		0.719662, 	-0.638761, 	0.677011, 	0.426932, 	-0.035015, 	-0.199777, 	-0.127390, 	0.209214, 
		-0.241407, 	0.394671, 	0.553515, 	-0.046183, 	0.299512, 	-0.853258, 	0.672052, 	0.064106, 
		0.352720, 	-0.224356, 	0.152166, 	0.756178, 	-0.800834, 	-0.717744, 	0.067581, 	-0.406055,
	};
//	ConvFilter:
	for(int filter = 0 ; filter < SEB_FC1_FILTER_OUT; filter++){
#pragma HLS PIPELINE off
		float sum = 0;
		for(int i = 0 ; i < SEB_FC1_FILTER_IN; i++){
#pragma HLS PIPELINE ii=4
			sum += inArr[i] * seb_fc1_weight[filter * SEB_FC1_FILTER_IN + i];
		}
		seb_fc1_out[filter] = relu(sum);
	}

//	//Fully connected 2
	for(int filter = 0 ; filter < SEB_FC2_FILTER_OUT; filter++){
#pragma HLS PIPELINE off
		float sum = 0;
		for(int i = 0 ; i < SEB_FC2_FILTER_IN; i++){
#pragma HLS PIPELINE ii=4
			sum += seb_fc1_out[i] * seb_fc2_weight[filter * SEB_FC2_FILTER_IN + i];
		}
		seb_fc2_out[filter] = sigmoid(sum);
	}


	for(int i = 0; i < CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_FILTER; i++){
		output.write(inArr[i] * seb_fc2_out[i]);
	}
	return;
}

void conv_4_1_accel(hls::stream<float>& input, hls::stream<float>& output){
	float bias[2] = {
		-0.045668,
		0.245635,
	};
	static float weights[64] = {
			1.786116,

			1.840739,

			-2.596637,

			-1.624576,

			-2.864458,

			-3.288004,

			1.430085,

			1.316417,

			-1.666196,

			1.676932,

			1.404078,

			-0.962405,

			1.620763,

			0.957052,

			2.557781,

			1.961144,

			0.829546,

			-2.698360,

			-2.391731,

			2.079472,

			-0.874036,

			1.750874,

			1.489102,

			-1.378578,

			1.460910,

			1.317545,

			1.051442,

			3.436466,

			-1.856105,

			1.260481,

			1.623804,

			3.074553,


			-2.290937,

			-2.109358,

			2.727872,

			1.550850,

			3.431461,

			3.070626,

			-1.428097,

			-1.287600,

			0.996867,

			-1.464423,

			-0.717010,

			0.799364,

			-1.274244,

			-1.072109,

			-2.917797,

			-2.359457,

			-1.124873,

			3.018427,

			2.333781,

			-1.824138,

			1.410056,

			-1.963489,

			-1.302895,

			1.272629,

			-1.870751,

			-1.298421,

			-1.247557,

			-3.649698,

			1.911782,

			-1.195884,

			-2.104844,

			-2.533237,
	};
#pragma HLS BIND_STORAGE variable=weights type=ram_s2p impl=lutram
//#pragma HLS ARRAY_PARTITION variable=bias type=complete

    float sum = 0;
    int in_offset = 0;
    int weight_offset = 0;
    int data_offset = 0;

    float out_conv3[CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_FILTER] = {0};
    stream2arr(input, out_conv3, CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_FILTER);

	// Convolution layer
    ConvFilter:
	for (int filter = 0; filter < CONV4_1_FILTER; filter++) {
#pragma HLS PIPELINE off
		ConvY:
		for (int ify = 0; ify < CONV4_1_OUT_SIZE; ify++) {
#pragma HLS PIPELINE off=true
			ConvX:
			for (int ifx = 0; ifx < CONV4_1_OUT_SIZE; ifx++){
#pragma HLS PIPELINE off=true
				ConvChannel:
				for (int inChan = 0; inChan < CONV3_FILTER; inChan++) {
#pragma HLS PIPELINE off=true
					ConvKy:
					for (int ky = 0; ky < CONV4_1_SIZE; ky++) {
#pragma HLS PIPELINE off=true
						ConvKx:
						for (int kx = 0; kx < CONV4_1_SIZE; kx++) {
#pragma HLS PIPELINE off=true
							in_offset = inChan * CONV3_OUT_SIZE * CONV3_OUT_SIZE + CONV3_OUT_SIZE * (ify + ky) + (ifx + kx);
							weight_offset = CONV4_1_SIZE * CONV4_1_SIZE * inChan + CONV4_1_IN_CHANNEL * CONV4_1_SIZE * CONV4_1_SIZE * filter + CONV4_1_SIZE * ky + kx;
							sum += out_conv3[in_offset] * weights[weight_offset];
						}
					}
				}
				output.write(sum + bias[filter]);
				sum = 0;
			}
		}
	}

	return;
	
}

void conv_4_2_accel(hls::stream<float>& input, hls::stream<float>& output){
	float bias[4] = {
		0.033004,
		-0.016164,
		-0.025241,
		0.025638,
	};
	static float weights[128] = {
			0.056688,

			0.027034,

			-0.002843,

			0.053451,

			0.011439,

			-0.037107,

			0.021848,

			0.000165,

			-0.002702,

			-0.004405,

			-0.050617,

			0.022091,

			-0.044167,

			0.030776,

			-0.071847,

			0.029567,

			0.014937,

			0.014401,

			0.017060,

			0.009996,

			-0.010538,

			0.000510,

			-0.007361,

			-0.005585,

			0.007118,

			0.003250,

			-0.014108,

			-0.002878,

			0.033414,

			-0.029297,

			0.013694,

			-0.014046,


			0.018023,

			-0.065542,

			0.001315,

			-0.040749,

			-0.009532,

			0.030033,

			-0.006455,

			-0.015911,

			-0.011943,

			-0.022642,

			-0.001370,

			-0.004919,

			-0.011567,

			-0.003687,

			0.057394,

			0.051565,

			-0.011507,

			-0.027392,

			0.016115,

			-0.076613,

			-0.022236,

			0.034299,

			-0.046714,

			-0.007831,

			-0.011844,

			-0.004353,

			-0.128827,

			-0.027479,

			-0.008606,

			0.006337,

			0.017345,

			-0.005308,


			-0.055310,

			-0.039515,

			0.009668,

			-0.014526,

			-0.005746,

			0.002012,

			0.016535,

			0.006442,

			-0.003820,

			0.002640,

			-0.021958,

			0.040884,

			0.000820,

			0.019054,

			-0.013808,

			-0.031312,

			-0.011571,

			-0.022510,

			-0.011442,

			-0.009164,

			-0.009617,

			0.056863,

			-0.001502,

			-0.016250,

			0.002493,

			0.001328,

			-0.042623,

			-0.042138,

			0.073445,

			0.008697,

			0.002470,

			0.020885,


			0.018054,

			0.090855,

			0.021407,

			-0.046416,

			0.017088,

			-0.047528,

			-0.017074,

			0.017688,

			-0.031950,

			0.041691,

			-0.010760,

			-0.046942,

			0.045132,

			0.057583,

			-0.074348,

			-0.034085,

			-0.041555,

			0.016329,

			-0.023874,

			0.034134,

			-0.012101,

			-0.033818,

			0.007565,

			0.013039,

			0.036650,

			0.006373,

			0.025313,

			0.014868,

			0.019727,

			-0.006669,

			0.017164,

			0.018955,
	};
#pragma HLS BIND_STORAGE variable=weights type=ram_s2p impl=lutram

    float sum = 0;
    int in_offset = 0;
    int weight_offset = 0;
    int data_offset = 0;

    float out_conv3[CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_FILTER] = {0};
    stream2arr(input, out_conv3, CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_FILTER);

	// Convolution layer
    ConvFilter:
	for (int filter = 0; filter < CONV4_2_FILTER; filter++) {
#pragma HLS PIPELINE off
		ConvY:
		for (int ify = 0; ify < CONV4_2_OUT_SIZE; ify++) {
#pragma HLS PIPELINE off
			ConvX:
			for (int ifx = 0; ifx < CONV4_2_OUT_SIZE; ifx++){
#pragma HLS PIPELINE off
				ConvChannel:
				for (int inChan = 0; inChan < CONV3_FILTER; inChan++) {
#pragma HLS PIPELINE off
					ConvKy:
					for (int ky = 0; ky < CONV4_2_SIZE; ky++) {
#pragma HLS PIPELINE off
						ConvKx:
						for (int kx = 0; kx < CONV4_2_SIZE; kx++) {
#pragma HLS PIPELINE off
							in_offset = inChan * CONV3_OUT_SIZE * CONV3_OUT_SIZE + CONV3_OUT_SIZE * (ify + ky) + (ifx + kx);
							weight_offset = CONV4_2_SIZE * CONV4_2_SIZE * inChan + CONV4_2_IN_CHANNEL * CONV4_2_SIZE * CONV4_2_SIZE * filter + CONV4_2_SIZE * ky + kx;
							sum += out_conv3[in_offset] * weights[weight_offset];
						}
					}
				}
				output.write(sum + bias[filter]);
				sum = 0;
			}
		}
	}
	return;
}
void duplicateArray(float input[], float output1[], float output2[], const int inSize){
	for(int i = 0; i < inSize; i++){
		output1[i] = input[i];
		output2[i] = input[i];
	}
	return;
}



void pnet_accel(float* input, float* output1, float* output2){
#pragma HLS INTERFACE mode=s_axilite port=return
#pragma HLS INTERFACE mode=m_axi port=input offset=slave bundle=gmem0 depth = INPUT_SIZE * INPUT_SIZE * CONV1_IN_CHANNEL
#pragma HLS INTERFACE mode=m_axi port=output1 offset=slave bundle=gmem1 depth = CONV4_1_OUT_SIZE * CONV4_1_OUT_SIZE * CONV4_1_FILTER
#pragma HLS INTERFACE mode=m_axi port=output2 offset=slave bundle=gmem2 depth = CONV4_2_OUT_SIZE * CONV4_2_OUT_SIZE * CONV4_2_FILTER
#pragma HLS DATAFLOW

	hls::stream<float> inStream;
	hls::stream<float> out1;
	hls::stream<float> out2;
	hls::stream<float> out3;
	hls::stream<float> out4;
	hls::stream<float> out4_1;
	hls::stream<float> out4_2;
	hls::stream<float> seb_out1;
	hls::stream<float> seb_out2;
	hls::stream<float> out5;
	hls::stream<float> out6;
	float out4arr[CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_FILTER];
	float out4arr_1[CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_FILTER];
	float out4arr_2[CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_FILTER];

#pragma HLS BIND_STORAGE variable=out4arr type=ram_s2p impl=lutram
#pragma HLS BIND_STORAGE variable=out4arr_1 type=ram_s2p impl=lutram
#pragma HLS BIND_STORAGE variable=out4arr_2 type=ram_s2p impl=lutram
	arr2stream(input, inStream, INPUT_SIZE * INPUT_SIZE * CONV1_IN_CHANNEL);
    conv_1_accel(inStream, out1);

    mp_1_accel(out1, out2);
//    printMat(out2, CONV1_FILTER, POOL1_OUT_SIZE);



    conv_2_accel(out2, out3);
//    printMat(out3, CONV2_FILTER, CONV2_OUT_SIZE);
    conv_3_accel(out3, out4);

    stream2arr(out4, out4arr, CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_FILTER);
    duplicateArray(out4arr, out4arr_1, out4arr_2, CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_FILTER);
    arr2stream(out4arr_1, out4_1, CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_FILTER);
    arr2stream(out4arr_2, out4_2, CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_FILTER);

	SEBlock(out4_1, seb_out1);
	SEBlock(out4_2, seb_out2);
//	printMat(seb_out1, CONV3_FILTER, CONV3_OUT_SIZE);

    conv_4_1_accel(seb_out1, out5);
    conv_4_2_accel(seb_out2, out6);

    stream2arr(out5, output1, CONV4_1_OUT_SIZE * CONV4_1_OUT_SIZE * CONV4_1_FILTER);
    stream2arr(out6, output2, CONV4_2_OUT_SIZE * CONV4_2_OUT_SIZE * CONV4_2_FILTER);

    return;
}
