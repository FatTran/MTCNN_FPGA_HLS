#include "config/onet_config.h"
float prelu(float input, float alpha){
    return (input >= 0) ? input : alpha * input;
}
void mp_1_accel(float* input, /*float* weights,*/ float* output){
//#pragma HLS INLINE off=false
// 	float weights[864] = {
// 		0.481470, 0.453283, 0.212088, 0.302011, -0.090088, -0.498306, -0.062487, -0.360821, -0.371171, 0.469717, 0.443920, 0.166898, 0.298781, -0.090504, -0.503422, -0.087253, -0.372925, -0.357667, 0.331012, 0.362157, 0.137930, 0.221912, -0.077655, -0.420072, -0.109269, -0.298550, -0.202567, -0.220768, -0.058986, 0.096379, -0.194155, 0.009529, 0.163006, -0.156467, -0.034742, 0.102943, -0.178678, -0.015580, 0.154033, -0.132280, 0.074211, 0.230683, -0.103345, 0.033851, 0.170643, -0.172207, -0.021035, 0.132980, -0.138532, 0.055387, 0.206854, -0.114629, 0.018888, 0.150143, 0.411917, 0.145009, -0.275313, 0.489466, 0.035096, -0.441260, 0.381395, -0.141148, -0.496990, 0.327353, 0.106126, -0.294067, 0.413106, 0.019585, -0.406782, 0.329460, -0.129789, -0.423699, 0.262431, 0.083098, -0.221349, 0.342454, 0.010561, -0.321407, 0.259132, -0.126330, -0.319185, -0.123179, -0.191804, -0.143414, -0.224664, -0.322900, -0.157930, -0.094375, -0.192043, -0.100830, 0.104224, 0.040318, 0.077588, 0.027237, -0.067622, 0.077936, 0.120237, 0.012004, 0.088138, 0.168861, 0.117471, 0.136696, 0.130374, 0.047909, 0.170771, 0.217589, 0.114819, 0.174688, -0.154230, 0.168389, 0.452563, -0.334091, -0.028058, 0.267748, -0.377676, -0.185208, 0.092889, -0.102154, 0.174050, 0.418012, -0.255446, 0.009651, 0.256653, -0.306289, -0.151571, 0.079936, -0.077160, 0.150101, 0.343893, -0.227925, -0.013435, 0.193040, -0.294643, -0.174836, 0.017955, 0.053334, 0.385849, 0.429701, -0.510157, -0.121509, 0.328221, -0.276099, -0.319128, 0.041481, 0.023120, 0.361417, 0.395427, -0.502519, -0.117221, 0.311507, -0.225968, -0.290282, 0.040645, 0.021460, 0.300909, 0.287958, -0.423666, -0.116639, 0.230430, -0.092851, -0.235158, -0.001835, -0.238095, -0.242411, -0.236886, -0.019197, -0.003783, -0.024408, 0.183948, 0.218650, 0.168339, -0.189956, -0.195047, -0.202152, 0.016877, 0.041432, 0.014997, 0.222661, 0.260250, 0.215762, -0.179134, -0.200460, -0.200526, -0.008002, 0.003797, -0.018624, 0.189378, 0.214647, 0.180276, -0.431066, 0.071126, 0.450887, -0.475691, 0.032071, 0.488466, -0.451223, 0.014313, 0.428503, -0.401145, 0.039228, 0.364519, -0.413135, 0.016231, 0.383155, -0.402355, -0.015272, 0.333539, -0.322876, 0.042371, 0.300935, -0.324560, 0.014831, 0.324477, -0.323810, -0.019677, 0.265728, -0.030762, -0.042107, -0.061602, -0.028655, -0.030333, -0.050883, -0.032384, -0.037319, -0.053764, 0.071605, 0.064629, 0.037763, 0.085518, 0.076753, 0.058250, 0.065177, 0.060635, 0.041795, 0.073127, 0.070666, 0.044321, 0.083434, 0.082327, 0.058895, 0.065124, 0.062745, 0.044993, -0.105525, -0.087094, -0.045488, -0.097331, -0.087410, -0.078053, -0.040573, -0.021526, -0.010481, -0.133531, -0.103012, -0.067800, -0.111872, -0.100855, -0.090861, -0.066369, -0.040306, -0.033532, -0.013752, 0.009105, 0.041282, 0.022532, 0.028134, 0.028167, 0.067058, 0.084912, 0.085490, -0.006875, 0.020345, 0.003189, 0.016256, 0.046781, 0.039055, 0.000756, 0.049267, 0.039718, 0.154536, 0.168805, 0.144902, 0.157584, 0.181179, 0.162514, 0.133011, 0.157198, 0.144916, -0.194601, -0.192153, -0.204836, -0.187775, -0.184437, -0.186847, -0.189502, -0.175885, -0.173175, -0.089744, -0.221047, -0.249862, -0.133544, -0.330033, -0.356159, -0.055831, -0.164284, -0.209514, 0.137317, 0.023289, -0.012846, 0.117046, -0.056751, -0.099503, 0.168183, 0.064704, 0.011129, 0.116206, 0.028137, 0.004994, 0.122882, -0.025044, -0.049028, 0.156613, 0.091472, 0.048046, 0.158977, 0.146491, 0.134560, 0.119721, 0.112027, 0.094478, 0.065249, 0.052093, 0.030686, 0.063935, 0.059100, 0.051287, -0.000129, 0.004829, -0.011199, -0.064490, -0.074887, -0.096881, 0.029654, 0.016817, 0.004096, -0.036141, -0.033691, -0.048712, -0.105273, -0.120401, -0.136689, -0.433561, -0.390903, 0.046282, -0.409809, 0.002378, 0.432147, 0.031516, 0.332648, 0.417384, -0.377987, -0.356159, 0.062574, -0.394137, 0.006257, 0.416272, -0.009231, 0.276279, 0.339144, -0.307329, -0.309952, 0.052026, -0.338754, 0.036422, 0.389295, -0.033456, 0.234174, 0.280130, 0.074780, 0.070968, 0.069850, 0.069527, 0.072602, 0.072090, 0.044203, 0.047759, 0.039663, 0.050616, 0.061699, 0.062314, 0.052184, 0.064613, 0.063914, 0.007110, 0.019822, 0.019572, 0.030738, 0.052623, 0.051244, 0.038162, 0.062685, 0.057295, -0.002295, 0.010575, 0.015739, 0.134767, 0.139699, 0.122801, -0.056269, -0.083514, -0.018125, 0.093613, 0.194522, 0.112622, -0.072146, -0.089430, -0.126266, -0.364628, -0.407461, -0.350662, -0.185964, -0.101780, -0.192029, 0.193975, 0.191036, 0.157774, 0.010725, -0.020485, 0.040406, 0.194409, 0.298011, 0.199856, -0.173521, -0.167793, -0.169017, -0.018536, 0.006439, -0.009854, 0.123307, 0.162400, 0.112991, -0.092357, -0.089717, -0.093783, 0.043755, 0.071367, 0.052654, 0.182060, 0.214431, 0.174068, -0.155978, -0.168299, -0.168291, -0.042798, -0.025367, -0.041487, 0.095955, 0.123218, 0.090035, -0.178866, -0.206262, -0.152185, -0.144496, -0.078761, -0.037792, -0.035654, -0.060959, -0.142712, 0.231564, 0.204773, 0.251314, 0.193910, 0.259873, 0.296399, 0.236029, 0.211029, 0.124808, -0.080449, -0.098847, -0.049515, -0.116577, -0.046561, -0.003597, -0.075363, -0.092755, -0.168966, -0.069940, -0.063260, -0.065753, -0.023693, -0.020322, -0.014779, -0.005997, -0.001838, -0.004690, 0.006851, 0.018584, 0.017092, 0.043932, 0.053820, 0.050052, 0.057501, 0.054121, 0.052785, 0.025133, 0.036284, 0.039717, 0.059073, 0.071908, 0.073597, 0.068506, 0.067833, 0.061910, 0.188871, 0.152256, 0.103371, 0.225393, 0.192270, 0.146588, 0.242101, 0.220853, 0.179367, -0.197312, -0.215054, -0.227682, -0.196026, -0.229385, -0.243979, -0.154716, -0.171268, -0.187996, 0.003109, -0.008335, -0.010232, 0.001435, -0.019601, -0.023179, 0.030006, 0.015797, 0.006873, 0.464682, 0.456599, 0.530192, -0.023538, 0.058673, 0.160351, -0.586131, -0.592317, -0.475410, 0.413070, 0.402304, 0.456070, -0.024728, 0.070875, 0.140938, -0.521221, -0.518722, -0.414825, 0.310464, 0.307105, 0.327683, -0.025384, 0.072991, 0.115770, -0.400375, -0.394058, -0.330359, -0.130112, -0.207411, -0.152235, 0.249748, 0.320043, 0.248386, -0.086941, -0.202854, -0.057125, -0.104409, -0.165199, -0.125572, 0.294087, 0.383884, 0.282477, -0.158833, -0.262872, -0.139705, -0.146840, -0.185815, -0.166573, 0.282208, 0.387812, 0.264363, -0.119631, -0.207409, -0.105480, 0.035980, 0.009748, 0.004658, -0.032565, -0.074613, -0.092977, 0.012588, -0.023100, -0.034745, -0.042220, -0.066293, -0.075292, -0.106656, -0.146509, -0.157499, -0.059326, -0.092227, -0.106266, 0.006768, -0.011928, -0.020540, -0.040576, -0.062746, -0.077770, 0.015741, -0.010512, -0.017165, 0.296574, 0.291548, 0.209383, -0.049073, -0.059021, -0.103979, -0.352552, -0.379846, -0.379003, 0.325896, 0.336670, 0.271449, 0.047667, 0.069148, 0.031317, -0.212277, -0.219078, -0.220822, 0.278032, 0.298331, 0.245722, 0.004873, 0.027311, -0.005300, -0.262515, -0.271204, -0.269290, -0.102523, -0.058697, 0.001340, -0.093140, -0.067644, 0.009612, -0.036230, -0.018456, 0.043265, -0.040988, 0.002245, 0.053189, -0.026666, 0.000156, 0.062170, 0.013237, 0.038726, 0.091134, -0.047070, -0.005622, 0.035837, -0.035850, -0.005973, 0.049531, -0.001191, 0.024650, 0.061505, -0.161152, -0.138822, -0.080384, -0.191187, -0.156931, -0.100290, -0.110510, -0.071705, -0.025735, -0.112991, -0.079511, -0.031865, -0.131283, -0.097778, -0.036845, -0.064360, -0.023923, 0.019106, -0.018984, 0.011440, 0.053941, -0.019107, 0.016948, 0.060424, 0.043183, 0.085175, 0.120500, -0.514108, -0.579939, -0.546107, 0.042971, 0.039895, 0.030344, 0.508324, 0.513949, 0.550220, -0.475105, -0.523864, -0.490109, 0.043319, 0.047334, 0.027416, 0.458333, 0.448328, 0.479630, -0.410604, -0.432783, -0.413195, 0.032233, 0.060094, 0.036122, 0.347044, 0.357007, 0.369311, -0.097394, -0.071932, -0.047685, -0.109547, -0.098716, -0.069781, -0.146534, -0.154708, -0.133739, 0.068157, 0.114406, 0.140872, 0.052348, 0.078787, 0.105615, -0.002411, 0.004920, 0.016357, 0.045611, 0.085523, 0.118256, 0.025358, 0.064344, 0.088968, -0.025402, -0.015927, 0.004526, 0.141100, -0.385697, -0.507434, 0.476272, -0.001971, -0.482362, 0.425918, 0.320762, -0.065721, 0.141368, -0.328675, -0.396693, 0.452744, 0.022736, -0.407981, 0.353687, 0.281306, -0.057412, 0.129611, -0.295335, -0.327344, 0.424457, 0.046659, -0.353233, 0.276254, 0.228893, -0.081087, 0.348063, 0.048026, -0.287834, 0.274213, -0.007343, -0.319283, 0.140240, -0.114344, -0.336954, 0.371329, 0.108676, -0.198846, 0.309278, 0.061075, -0.225169, 0.165469, -0.058835, -0.256583, 0.314225, 0.083120, -0.194072, 0.257594, 0.033799, -0.227742, 0.110032, -0.092812, -0.275228, -0.089456, -0.083230, -0.058763, -0.093577, -0.076198, -0.066280, -0.096558, -0.081777, -0.070243, -0.004770, 0.000477, 0.018851, -0.016122, 0.000584, 0.016153, -0.021633, -0.010569, 0.006807, 0.027583, 0.035095, 0.053186, 0.026817, 0.043785, 0.048847, 0.021596, 0.030968, 0.039394, -0.031871, -0.017430, -0.024776, -0.030837, -0.029525, -0.027380, -0.030739, -0.030829, -0.034820, 0.053715, 0.067465, 0.063723, 0.048746, 0.052764, 0.048575, 0.044105, 0.046854, 0.046004, 0.051938, 0.064877, 0.064780, 0.052371, 0.054605, 0.052213, 0.048345, 0.044814, 0.047899
// 	};
//     static float bias[32] = {
// 		0.091005,
// 		-0.180599,
// 		0.124300,
// 		0.644636,
// 		-0.195649,
// 		0.048859,
// 		-0.194113,
// 		0.133441,
// 		0.001896,
// 		-0.096055,
// 		-0.107946,
// 		-0.487616,
// 		-0.257139,
// 		0.056912,
// 		-0.123677,
// 		-0.008434,
// 		-0.240890,
// 		0.368883,
// 		-0.238203,
// 		-0.079689,
// 		0.119158,
// 		-0.003842,
// 		-0.487672,
// 		-0.319927,
// 		0.011212,
// 		-0.442355,
// 		0.115772,
// 		0.103502,
// 		0.089776,
// 		-0.229880,
// 		-0.092763,
// 		-0.335248
// 	};

// 	static float prelu_weight[32] = {
// 		0.060191,
// 		-0.000546,
// 		0.118366,
// 		0.367750,
// 		-0.470247,
// 		0.068185,
// 		-0.116294,
// 		0.190940,
// 		-0.020070,
// 		0.014685,
// 		0.012389,
// 		0.647218,
// 		0.141163,
// 		0.256490,
// 		0.011165,
// 		-0.143655,
// 		-0.000181,
// 		-0.242154,
// 		0.284636,
// 		-0.060145,
// 		0.250575,
// 		0.000014,
// 		0.018780,
// 		-0.159719,
// 		0.030098,
// 		0.232222,
// 		0.313915,
// 		-0.010857,
// 		0.419000,
// 		-0.230015,
// 		-0.015717,
// 		0.359514
// 	};
// #pragma HLS ARRAY_PARTITION variable=bias type=complete
// #pragma HLS ARRAY_PARTITION variable=prelu_weight type=complete

//     float out_conv[CONV1_OUT_SIZE * CONV1_OUT_SIZE * CONV1_FILTER] = {0};
// #pragma HLS BIND_STORAGE variable=out_conv type=ram_s2p impl=auto
// //#pragma HLS ARRAY_RESHAPE variable=out_conv type=cycle dim=1 factor=CONV1_OUT_SIZE * CONV1_OUT_SIZE
//     float sum = 0;
//     int in_offset = 0;
//     int weight_offset = 0;
//     int data_offset = 0;

// 	// Convolution layer
//     ConvFilter:
// 	for (int filter = 0; filter < CONV1_FILTER; filter++) {
// #pragma HLS UNROLL off=true
// #pragma HLS PIPELINE off=true
// 		ConvY:
// 		for (int ify = 0; ify < CONV1_OUT_SIZE; ify++) {
// #pragma HLS PIPELINE off=true
// #pragma HLS UNROLL off=true
// 			ConvX:
// 			for (int ifx = 0; ifx < CONV1_OUT_SIZE; ifx++){
// #pragma HLS PIPELINE off=false ii=27
// #pragma HLS UNROLL off=true
// 				ConvChannel:
// 				for (int inChan = 0; inChan < CONV1_IN_CHANNEL; inChan++) {
// #pragma HLS UNROLL off=true
// #pragma HLS PIPELINE off=false ii=48
// 					ConvKy:
// 					for (int ky = 0; ky < CONV1_SIZE; ky++) {
// #pragma HLS UNROLL off=false
// #pragma HLS PIPELINE off=true
// //#pragma HLS PIPELINE off=true
// 						ConvKx:
// 						for (int kx = 0; kx < CONV1_SIZE; kx++) {
// #pragma HLS UNROLL off=false
// #pragma HLS PIPELINE off=true
// 							data_offset = CONV1_OUT_SIZE * CONV1_OUT_SIZE * filter + ify * CONV1_OUT_SIZE + ifx;
// 							in_offset = inChan * INPUT_SIZE * INPUT_SIZE + INPUT_SIZE * (ify + ky) + (ifx + kx);
// 							weight_offset = CONV1_SIZE * CONV1_SIZE * inChan + CONV1_IN_CHANNEL * CONV1_SIZE * CONV1_SIZE * filter + CONV1_SIZE * ky + kx;
// //							sum = sum + input[in_offset] * weights[weight_offset];
// 							out_conv[data_offset] += input[in_offset] * weights[weight_offset];
// 						}
// 					}
// 				}
// //				data_offset = CONV1_OUT_SIZE * CONV1_OUT_SIZE * filter + ify * CONV1_OUT_SIZE + ifx;
// //				out_conv[data_offset] = prelu(sum + bias[filter], prelu_weight[filter]);
// //				sum = 0;
// 			}
// 		}
// 	}

// 	for(int i = 0; i < CONV1_FILTER; i++){
// 		for(int j = 0 ; j < CONV1_OUT_SIZE * CONV1_OUT_SIZE; j++){
// 			int offset = i * CONV1_OUT_SIZE * CONV1_OUT_SIZE + j;
// 			int value = out_conv[offset];
// 			out_conv[offset] = prelu(value + bias[i], prelu_weight[i]);
// 		}
// 	}

	//Max pooling layer
	int in_offset = 0;
	Pool_y:
	for (int ify = 0; ify < POOL1_OUT_SIZE; ify++) {
#pragma HLS PIPELINE off=true
#pragma HLS UNROLL off=true
		Pool_x:
		for (int ifx = 0; ifx < POOL1_OUT_SIZE; ifx++) {
#pragma HLS PIPELINE off=true
#pragma HLS UNROLL off=true
			//float max = FLOAT_MIN;
			Pool_channel:
			for (int inChan = 0; inChan < CONV1_FILTER; inChan++) {
#pragma HLS PIPELINE off=true
#pragma HLS UNROLL off=true
				int data_offset = inChan * POOL1_OUT_SIZE * POOL1_OUT_SIZE + ify * POOL1_OUT_SIZE + ifx;
				int out_conv_offset = inChan * CONV1_OUT_SIZE * CONV1_OUT_SIZE + ify * STRIDE1 * CONV1_OUT_SIZE + ifx * STRIDE1;
				// float max = out_conv[out_conv_offset];
				float max = input[out_conv_offset];
				Ky:
				for (int ky = 0; ky < MP1_SIZE; ky++) {
#pragma HLS PIPELINE off=false ii=7
#pragma HLS UNROLL off=true
					Kx:
					for (int kx = 0; kx < MP1_SIZE; kx++) {
#pragma HLS UNROLL off=false
#pragma HLS PIPELINE off=true
						in_offset = out_conv_offset + ky * CONV1_OUT_SIZE + kx;
						if( ((CONV1_OUT_SIZE - ifx * STRIDE1) >= MP1_SIZE) && ((CONV1_OUT_SIZE - ify * STRIDE1) >= MP1_SIZE)){
							// max = (out_conv[in_offset] > max) ? out_conv[in_offset] : max;
							max = (input[in_offset] > max) ? input[in_offset] : max;
						} else {
							if(kx >= (CONV1_OUT_SIZE - ifx * STRIDE1)){
								in_offset = in_offset - 1;
							}
							if(ky >= (CONV1_OUT_SIZE - ify * STRIDE1)){
								in_offset = in_offset - CONV1_OUT_SIZE;
							}
							// max = (out_conv[in_offset] > max) ? out_conv[in_offset] : max;
							max = (input[in_offset] > max) ? input[in_offset] : max;
						}
					}
				}
				output[data_offset] = max;
			}
		}
	}
	return;
}

void conv_mp_2_accel(float* input, float* weights, float* output){
//#pragma HLS INLINE off=false
    static float bias[64] = {
		0.061261,
		-0.000578,
		0.037342,
		0.378127,
		0.244001,
		0.137335,
		0.069740,
		0.037474,
		0.223293,
		-0.096679,
		0.020467,
		-0.104246,
		-0.008275,
		0.067969,
		-0.003060,
		0.293353,
		0.228649,
		0.051284,
		-0.030087,
		0.002674,
		-0.054224,
		0.058736,
		0.032874,
		0.086688,
		-0.088989,
		0.160925,
		0.029371,
		-0.003945,
		0.024740,
		0.059425,
		-0.017150,
		-0.174602,
		-0.047898,
		0.020810,
		0.249534,
		-0.289985,
		-0.045498,
		-0.061481,
		-0.078131,
		-0.125643,
		0.035532,
		0.255421,
		0.025267,
		-0.003541,
		0.314979,
		0.004535,
		0.073697,
		-0.105432,
		0.025050,
		0.185940,
		0.098155,
		0.006137,
		0.030899,
		0.044182,
		0.020264,
		0.012993,
		-0.093881,
		-0.009478,
		0.008197,
		0.054467,
		0.044414,
		0.152096,
		-0.049346,
		0.109363
	};

	static float prelu_weight[64] = {
		-0.116740,
		0.013232,
		0.040346,
		0.419776,
		0.092613,
		0.033490,
		0.126992,
		0.146557,
		-0.013337,
		0.208791,
		0.055419,
		-0.003940,
		0.222108,
		0.178859,
		0.080410,
		0.000359,
		0.310233,
		0.095830,
		0.115588,
		-0.004398,
		-0.040722,
		0.128041,
		0.090602,
		0.139380,
		-0.003996,
		0.200090,
		-0.008777,
		0.000268,
		0.055488,
		0.064155,
		0.082735,
		-0.115343,
		0.053106,
		0.081828,
		0.115080,
		-0.036831,
		-0.018783,
		-0.062436,
		0.096450,
		-0.007399,
		-0.014200,
		0.234078,
		0.096615,
		-0.194867,
		0.443423,
		0.102765,
		0.052388,
		-0.074168,
		0.136888,
		0.095405,
		0.191027,
		0.017007,
		0.754263,
		0.038809,
		-0.006324,
		-0.124580,
		0.122514,
		0.091285,
		0.025713,
		-0.065574,
		0.090860,
		0.404067,
		0.080797,
		0.141770
	};
#pragma HLS ARRAY_PARTITION variable=bias type=complete
#pragma HLS ARRAY_PARTITION variable=prelu_weight type=complete

    float out_conv[CONV2_OUT_SIZE * CONV2_OUT_SIZE * CONV2_FILTER] = {0};
#pragma HLS BIND_STORAGE variable=out_conv type=ram_s2p impl=bram
    float sum = 0;
    int in_offset = 0;
    int weight_offset = 0;
    int data_offset = 0;

	// Convolution layer
    ConvFilter:
	for (int filter = 0; filter < CONV2_FILTER; filter++) {
#pragma HLS UNROLL off=true
#pragma HLS PIPELINE off=true
		ConvY:
		for (int ify = 0; ify < CONV2_OUT_SIZE; ify++) {
#pragma HLS PIPELINE off=true
#pragma HLS UNROLL off=true
			ConvX:
			for (int ifx = 0; ifx < CONV2_OUT_SIZE; ifx++){
#pragma HLS PIPELINE off=true
#pragma HLS UNROLL off=true
				ConvChannel:
				for (int inChan = 0; inChan < CONV1_FILTER; inChan++) {
#pragma HLS UNROLL off=true
#pragma HLS PIPELINE off=false ii=46
					ConvKy:
					for (int ky = 0; ky < CONV2_SIZE; ky++) {
#pragma HLS UNROLL off=true
#pragma HLS PIPELINE off=true
//#pragma HLS PIPELINE off=true
						ConvKx:
						for (int kx = 0; kx < CONV2_SIZE; kx++) {
#pragma HLS UNROLL off=false
#pragma HLS PIPELINE off=true
							data_offset = CONV2_OUT_SIZE * CONV2_OUT_SIZE * filter + ify * CONV2_OUT_SIZE + ifx;
							in_offset = inChan * POOL1_OUT_SIZE * POOL1_OUT_SIZE + POOL1_OUT_SIZE * (ify + ky) + (ifx + kx);
							weight_offset = CONV2_SIZE * CONV2_SIZE * inChan + CONV1_FILTER * CONV2_SIZE * CONV2_SIZE * filter + CONV2_SIZE * ky + kx;
//							sum = sum + input[in_offset] * weights[weight_offset];
							out_conv[data_offset] += input[in_offset] * weights[weight_offset];
						}
					}
				}
//				data_offset = CONV2_OUT_SIZE * CONV2_OUT_SIZE * filter + ify * CONV2_OUT_SIZE + ifx;
//				out_conv[data_offset] = prelu(sum + bias[filter], prelu_weight[filter]);
//				sum = 0;
			}
		}
	}

	for(int i = 0 ; i < CONV2_FILTER; i++){
		for(int j = 0 ; j < CONV2_OUT_SIZE * CONV2_OUT_SIZE; j++){
			int offset = i * CONV2_OUT_SIZE * CONV2_OUT_SIZE + j;
			out_conv[offset] = prelu(out_conv[offset] + bias[i], prelu_weight[i]);
		}
	}

	//Max pooling layer
	Pool_y:
	for (int ify = 0; ify < POOL2_OUT_SIZE; ify++) {
#pragma HLS PIPELINE off=true
#pragma HLS UNROLL off=true
		Pool_x:
		for (int ifx = 0; ifx < POOL2_OUT_SIZE; ifx++) {
#pragma HLS PIPELINE off=true
#pragma HLS UNROLL off=true
			//float max = FLOAT_MIN;
			Pool_channel:
			for (int inChan = 0; inChan < CONV2_FILTER; inChan++) {
#pragma HLS PIPELINE off=true
#pragma HLS UNROLL off=true
				int data_offset = inChan * POOL2_OUT_SIZE * POOL2_OUT_SIZE + ify * POOL2_OUT_SIZE + ifx;
				int out_conv_offset = inChan * CONV2_OUT_SIZE * CONV2_OUT_SIZE + ify * STRIDE2 * CONV2_OUT_SIZE + ifx * STRIDE2;
				float max = out_conv[out_conv_offset];
				Ky:
				for (int ky = 0; ky < MP2_SIZE; ky++) {
#pragma HLS PIPELINE off=false ii=6
#pragma HLS UNROLL off=true
					Kx:
					for (int kx = 0; kx < MP2_SIZE; kx++) {
#pragma HLS UNROLL off=false
#pragma HLS PIPELINE off=true
						in_offset = out_conv_offset + ky * CONV2_OUT_SIZE + kx;
						if( ((CONV2_OUT_SIZE - ifx * STRIDE2) >= MP2_SIZE) && ((CONV2_OUT_SIZE - ify * STRIDE2) >= MP2_SIZE)){
							max = (out_conv[in_offset] > max) ? out_conv[in_offset] : max;
						} else {
							if(kx >= (CONV2_OUT_SIZE - ifx * STRIDE2)){
								in_offset = in_offset - 1;
							}
							if(ky >= (CONV2_OUT_SIZE - ify * STRIDE2)){
								in_offset = in_offset - CONV2_OUT_SIZE;
							}
							max = (out_conv[in_offset] > max) ? out_conv[in_offset] : max;
						}
					}
				}
				output[data_offset] = max;
			}
		}
	}
	return;
}

void conv_mp_3_accel(float* input, float* weights, float* output){
// #pragma HLS INTERFACE mode=m_axi port=input offset=slave bundle=gmem0 depth = POOL2_OUT_SIZE * POOL2_OUT_SIZE * CONV2_FILTER
// #pragma HLS INTERFACE mode=m_axi port=weights offset=slave bundle=gmem1 depth = CONV3_SIZE * CONV3_SIZE * CONV3_FILTER * CONV2_FILTER
// #pragma HLS INTERFACE mode=m_axi port=output offset=slave bundle=gmem2 depth = POOL3_OUT_SIZE * POOL3_OUT_SIZE * CONV3_FILTER

    static float bias[64] = {
		-0.019625,
		-0.074124,
		0.033882,
		0.045653,
		0.043943,
		-0.016889,
		0.006075,
		-0.159429,
		0.017155,
		-0.007684,
		-0.012211,
		0.035061,
		0.061986,
		0.015760,
		-0.035924,
		-0.122323,
		0.095572,
		0.140349,
		-0.063840,
		0.118247,
		-0.087204,
		-0.120022,
		0.045455,
		-0.133270,
		0.129122,
		0.009667,
		-0.075481,
		-0.111734,
		-0.023959,
		0.018630,
		-0.017908,
		0.173411,
		-0.101973,
		0.060585,
		0.034107,
		0.039541,
		0.149334,
		0.092184,
		-0.012331,
		0.057473,
		-0.005152,
		-0.021428,
		-0.042585,
		-0.073133,
		-0.047674,
		-0.030265,
		0.040098,
		0.133464,
		0.026571,
		0.005048,
		0.078072,
		0.026129,
		0.036747,
		-0.106770,
		-0.052449,
		-0.032939,
		-0.063667,
		0.031933,
		0.024725,
		0.050516,
		-0.082087,
		0.073330,
		-0.030324,
		0.173951
	};

	static float prelu_weight[64] = {
		-0.247555,
		0.182025,
		0.149368,
		0.216678,
		0.247126,
		0.289428,
		0.146061,
		0.004253,
		0.140606,
		0.107569,
		0.067046,
		0.092393,
		0.126467,
		0.104691,
		-0.299055,
		-0.028616,
		0.275921,
		0.156686,
		0.200443,
		0.021560,
		0.023298,
		0.029530,
		0.012633,
		0.139846,
		0.257074,
		0.186679,
		0.526032,
		0.018798,
		0.071621,
		-0.160816,
		0.152596,
		-0.014185,
		0.063218,
		0.121433,
		0.172229,
		0.190983,
		0.221642,
		0.073288,
		0.134300,
		0.179205,
		0.166986,
		0.118765,
		0.084652,
		0.201109,
		0.164657,
		0.125340,
		0.118212,
		0.212340,
		0.208564,
		0.080113,
		-0.001785,
		0.137778,
		0.096727,
		0.072224,
		0.008331,
		0.138303,
		0.103221,
		0.098306,
		0.021123,
		0.195150,
		0.259347,
		0.133091,
		0.011789,
		0.134729
	};
#pragma HLS ARRAY_PARTITION variable=bias type=complete
#pragma HLS ARRAY_PARTITION variable=prelu_weight type=complete

    float out_conv[CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_FILTER] = {0};
    float sum = 0;
    int in_offset = 0;
    int weight_offset = 0;
    int data_offset = 0;

	// Convolution layer
    ConvFilter:
	for (int filter = 0; filter < CONV3_FILTER; filter++) {
#pragma HLS UNROLL off=true
#pragma HLS PIPELINE off=true
		ConvY:
		for (int ify = 0; ify < CONV3_OUT_SIZE; ify++) {
#pragma HLS PIPELINE off=true
#pragma HLS UNROLL off=true
			ConvX:
			for (int ifx = 0; ifx < CONV3_OUT_SIZE; ifx++){
#pragma HLS PIPELINE off=true
#pragma HLS UNROLL off=true
				ConvChannel:
				for (int inChan = 0; inChan < CONV2_FILTER; inChan++) {
#pragma HLS UNROLL off=true
#pragma HLS PIPELINE off=false ii=46
					ConvKy:
					for (int ky = 0; ky < CONV3_SIZE; ky++) {
#pragma HLS UNROLL off=true
#pragma HLS PIPELINE off=true
//#pragma HLS PIPELINE off=true
						ConvKx:
						for (int kx = 0; kx < CONV3_SIZE; kx++) {
#pragma HLS UNROLL off=false
#pragma HLS PIPELINE off=true
							data_offset = CONV3_OUT_SIZE * CONV3_OUT_SIZE * filter + ify * CONV3_OUT_SIZE + ifx;
							in_offset = inChan * POOL2_OUT_SIZE * POOL2_OUT_SIZE + POOL2_OUT_SIZE * (ify + ky) + (ifx + kx);
							weight_offset = CONV3_SIZE * CONV3_SIZE * inChan + CONV2_FILTER * CONV3_SIZE * CONV3_SIZE * filter + CONV3_SIZE * ky + kx;
//							sum = sum + input[in_offset] * weights[weight_offset];
//							data_offset = CONV2_OUT_SIZE * CONV2_OUT_SIZE * filter + ify * CONV2_OUT_SIZE + ifx;
							out_conv[data_offset] += input[in_offset] * weights[weight_offset];
						}
					}
				}
//				data_offset = CONV3_OUT_SIZE * CONV3_OUT_SIZE * filter + ify * CONV3_OUT_SIZE + ifx;
//				out_conv[data_offset] = prelu(sum + bias[filter], prelu_weight[filter]);
//				sum = 0;
			}
		}
	}

	for(int i = 0 ; i < CONV3_FILTER; i++){
		for(int j = 0 ; j < CONV3_OUT_SIZE * CONV3_OUT_SIZE; j++){
			int offset = i * CONV3_OUT_SIZE * CONV3_OUT_SIZE + j;
			out_conv[offset] = prelu(out_conv[offset] + bias[i], prelu_weight[i]);
		}
	}

	//Max pooling layer
	Pool_y:
	for (int ify = 0; ify < POOL3_OUT_SIZE; ify++) {
#pragma HLS PIPELINE off=true
#pragma HLS UNROLL off=true
		Pool_x:
		for (int ifx = 0; ifx < POOL3_OUT_SIZE; ifx++) {
#pragma HLS PIPELINE off=true
#pragma HLS UNROLL off=true
			//float max = FLOAT_MIN;
			Pool_channel:
			for (int inChan = 0; inChan < CONV3_FILTER; inChan++) {
#pragma HLS PIPELINE off=true
#pragma HLS UNROLL off=true
				int data_offset = inChan * POOL3_OUT_SIZE * POOL3_OUT_SIZE + ify * POOL3_OUT_SIZE + ifx;
				int out_conv_offset = inChan * CONV3_OUT_SIZE * CONV3_OUT_SIZE + ify * STRIDE3 * CONV3_OUT_SIZE + ifx * STRIDE3;
				float max = out_conv[out_conv_offset];
				Ky:
				for (int ky = 0; ky < MP3_SIZE; ky++) {
#pragma HLS PIPELINE off=false ii=6
#pragma HLS UNROLL off=true
					Kx:
					for (int kx = 0; kx < MP3_SIZE; kx++) {
#pragma HLS UNROLL off=false
#pragma HLS PIPELINE off=true
						in_offset = out_conv_offset + ky * CONV3_OUT_SIZE + kx;
						if( ((CONV3_OUT_SIZE - ifx * STRIDE3) >= MP3_SIZE) && ((CONV3_OUT_SIZE - ify * STRIDE3) >= MP3_SIZE)){
							max = (out_conv[in_offset] > max) ? out_conv[in_offset] : max;
						} else {
							if(kx >= (CONV3_OUT_SIZE - ifx * STRIDE3)){
								in_offset = in_offset - 1;
							}
							if(ky >= (CONV3_OUT_SIZE - ify * STRIDE3)){
								in_offset = in_offset - CONV3_OUT_SIZE;
							}
							max = (out_conv[in_offset] > max) ? out_conv[in_offset] : max;
						}
					}
				}
				output[data_offset] = max;
			}
		}
	}
	return;
	
}

void conv_4_accel(float* input, float* weights, float* output){
// #pragma HLS INTERFACE mode=m_axi port=input offset=slave bundle=gmem0 depth = POOL3_OUT_SIZE * POOL3_OUT_SIZE *	CONV3_FILTER 
// #pragma HLS INTERFACE mode=m_axi port=weights offset=slave bundle=gmem1 depth = CONV4_SIZE * CONV4_SIZE * CONV4_FILTER * CONV3_FILTER
// #pragma HLS INTERFACE mode=m_axi port=output offset=slave bundle=gmem2 depth = CONV4_OUT_SIZE * CONV4_OUT_SIZE * CONV4_FILTER
//#pragma HLS DATAFLOW
	static float bias[128] = {
		0.170342,
		0.049385,
		0.124261,
		0.068352,
		0.051676,
		0.041113,
		0.031634,
		-0.004532,
		0.103248,
		0.136028,
		0.056100,
		0.099352,
		0.095258,
		0.013900,
		0.010728,
		0.036785,
		0.130742,
		-0.044804,
		-0.048151,
		0.051982,
		-0.049419,
		0.089090,
		0.149931,
		0.195629,
		0.097145,
		0.132458,
		-0.031072,
		0.047029,
		-0.037322,
		0.027881,
		-0.122090,
		0.092622,
		0.066258,
		0.029142,
		-0.017017,
		-0.044342,
		0.017957,
		0.080944,
		0.062235,
		0.210846,
		0.027445,
		0.103211,
		0.011335,
		0.073760,
		-0.008697,
		-0.033748,
		0.010067,
		0.094885,
		0.045748,
		0.009396,
		0.055366,
		0.029122,
		0.031661,
		0.112958,
		0.161523,
		-0.013122,
		0.047818,
		0.074246,
		0.001481,
		0.123454,
		-0.049236,
		0.021587,
		0.051583,
		0.051421,
		0.136585,
		0.077225,
		0.176791,
		0.139140,
		0.017440,
		0.068762,
		0.085450,
		0.201731,
		0.073872,
		0.086883,
		0.030477,
		-0.203255,
		0.120054,
		0.135873,
		0.004726,
		0.052258,
		0.168768,
		0.183855,
		0.041383,
		0.149999,
		0.130170,
		0.161985,
		0.047518,
		0.099023,
		0.049986,
		0.024346,
		0.035408,
		0.104178,
		0.067168,
		-0.276582,
		0.027736,
		0.164293,
		-0.012571,
		-0.003078,
		-0.007316,
		0.190991,
		-0.127901,
		0.110348,
		0.079045,
		0.101662,
		0.076269,
		0.049888,
		0.000523,
		0.062666,
		0.127152,
		0.053026,
		0.217277,
		-0.023054,
		0.038793,
		0.019673,
		0.119944,
		0.080668,
		0.114570,
		0.117600,
		-0.041556,
		0.139563,
		0.046911,
		0.051148,
		-0.001828,
		0.006467,
		0.035395,
		-0.020264,
		-0.036804,
		0.164348
	};

	static float prelu_weight[128] = {
		-0.134727,
		0.005563,
		-0.160689,
		-0.038867,
		-0.079676,
		-0.178053,
		-0.210070,
		-0.068302,
		-0.172901,
		-0.092908,
		-0.097165,
		-0.112246,
		-0.081404,
		-0.027271,
		-0.127477,
		-0.214579,
		0.001307,
		-0.214217,
		-0.149456,
		-0.066305,
		0.051153,
		-0.099446,
		-0.037601,
		-0.069578,
		-0.046829,
		-0.111529,
		-0.091064,
		-0.129426,
		-0.063909,
		-0.203795,
		-0.172494,
		-0.007764,
		-0.025396,
		-0.043960,
		-0.142101,
		-0.023265,
		-0.022188,
		-0.157226,
		-0.089001,
		-0.070103,
		-0.180397,
		-0.021729,
		-0.042317,
		0.032936,
		0.064194,
		-0.191313,
		-0.240193,
		-0.056270,
		-0.101154,
		-0.018170,
		-0.067626,
		-0.039679,
		-0.027156,
		0.005601,
		-0.130776,
		-0.161159,
		-0.106803,
		-0.240786,
		-0.179629,
		0.001981,
		-0.316591,
		-0.199813,
		-0.180721,
		-0.081404,
		-0.014598,
		0.023632,
		-0.089236,
		0.048687,
		-0.187610,
		-0.108606,
		-0.122679,
		-0.041955,
		0.040413,
		-0.011745,
		0.026942,
		-0.242993,
		0.007918,
		-0.191622,
		-0.043090,
		-0.022405,
		-0.076690,
		0.049179,
		-0.059070,
		-0.023123,
		-0.159015,
		0.041029,
		-0.159324,
		0.064561,
		-0.111026,
		-0.098444,
		-0.133230,
		-0.069898,
		-0.125146,
		-0.024524,
		-0.029373,
		-0.035208,
		-0.111038,
		-0.151964,
		-0.133614,
		-0.151040,
		-0.282987,
		-0.195675,
		0.010577,
		-0.089837,
		-0.010649,
		-0.007043,
		-0.092780,
		-0.145788,
		0.024348,
		-0.070834,
		-0.109305,
		-0.158703,
		0.047930,
		-0.159494,
		0.003426,
		-0.051509,
		-0.002145,
		-0.060658,
		-0.118554,
		0.068419,
		0.029736,
		-0.139105,
		-0.040739,
		-0.168193,
		-0.034549,
		-0.086859,
		-0.020808,
		0.094435
	};
#pragma HLS ARRAY_PARTITION variable=bias type=complete
#pragma HLS ARRAY_PARTITION variable=prelu_weight type=complete

    float sum = 0;
    int in_offset = 0;
    int weight_offset = 0;
    int data_offset = 0;

	// Convolution layer
    ConvFilter:
	for (int filter = 0; filter < CONV4_FILTER; filter++) {
#pragma HLS UNROLL off=true
#pragma HLS PIPELINE off=true
		ConvY:
		for (int ify = 0; ify < CONV4_OUT_SIZE; ify++) {
#pragma HLS PIPELINE off=true
#pragma HLS UNROLL off=true
			ConvX:
			for (int ifx = 0; ifx < CONV4_OUT_SIZE; ifx++){
#pragma HLS PIPELINE off=true
#pragma HLS UNROLL off=true
				ConvChannel:
				for (int inChan = 0; inChan < CONV4_IN_CHANNEL; inChan++) {
#pragma HLS UNROLL off=true
#pragma HLS PIPELINE off=false ii=21
					ConvKy:
					for (int ky = 0; ky < CONV4_SIZE; ky++) {
#pragma HLS UNROLL off=true
#pragma HLS PIPELINE off=true
//#pragma HLS PIPELINE off=true
						ConvKx:
						for (int kx = 0; kx < CONV4_SIZE; kx++) {
#pragma HLS UNROLL off=false
#pragma HLS PIPELINE off=true
							data_offset = CONV4_OUT_SIZE * CONV4_OUT_SIZE * filter + ify * CONV4_OUT_SIZE + ifx;
							in_offset = inChan * POOL3_OUT_SIZE * POOL3_OUT_SIZE + POOL3_OUT_SIZE * (ify + ky) + (ifx + kx);
							weight_offset = CONV4_SIZE * CONV4_SIZE * inChan + CONV4_IN_CHANNEL * CONV4_SIZE * CONV4_SIZE * filter + CONV4_SIZE * ky + kx;
//							sum = sum + input[in_offset] * weights[weight_offset];
							output[data_offset] += input[in_offset] * weights[weight_offset];
						}
					}
				}
//				data_offset = CONV4_OUT_SIZE * CONV4_OUT_SIZE * filter + ify * CONV4_OUT_SIZE + ifx;
//				output[data_offset] = prelu(sum + bias[filter], prelu_weight[filter]);
//				sum = 0;
			}
		}
	}

	for(int i = 0 ; i < CONV4_FILTER; i++){
		for(int j = 0 ; j < CONV4_OUT_SIZE * CONV4_OUT_SIZE; j++){
			int offset = i * CONV4_OUT_SIZE * CONV4_OUT_SIZE + j;
			output[offset] = prelu(output[offset] + bias[i], prelu_weight[i]);
		}
	}

	return;
	
}

void flatten_accel(float* input, float* output){
	// Flatten and transpose matrix
// #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=CONV4_OUT_SIZE * CONV4_OUT_SIZE * CONV4_FILTER
// #pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem1 depth=CONV4_OUT_SIZE * CONV4_OUT_SIZE * CONV4_FILTER
	for(int filter = 0; filter < CONV4_FILTER; filter++){
		for(int y = 0 ; y < CONV4_OUT_SIZE; y++){
			for(int x = 0 ; x < CONV4_OUT_SIZE; x++){
				int out_offset = filter * CONV4_OUT_SIZE * CONV4_OUT_SIZE + x * CONV4_OUT_SIZE + y;
				int in_offset = filter * CONV4_OUT_SIZE * CONV4_OUT_SIZE + y * CONV4_OUT_SIZE + x;
				output[out_offset] = input[in_offset];			
			}
		}
	}
	return;

}

void dense_1_accel(float* input, float* weights, float* output){
// #pragma HLS INTERFACE mode=m_axi port=input offset=slave bundle=gmem0 depth = CONV4_OUT_SIZE * CONV4_OUT_SIZE *	CONV4_FILTER 
// #pragma HLS INTERFACE mode=m_axi port=weights offset=slave bundle=gmem1 depth = CONV4_OUT_SIZE * CONV4_OUT_SIZE * CONV4_FILTER * FC1_DENSE_SIZE
// #pragma HLS INTERFACE mode=m_axi port=output offset=slave bundle=gmem2 depth = FC1_DENSE_SIZE
//#pragma HLS DATAFLOW
	static float bias[256] = {
		0.102012,
		0.033121,
		0.023577,
		0.081617,
		0.133923,
		0.092727,
		0.138967,
		-0.055601,
		0.021363,
		0.080380,
		0.092952,
		0.028567,
		-0.007645,
		0.087047,
		0.063298,
		0.065645,
		-0.136482,
		0.096905,
		0.063317,
		-0.002610,
		0.063645,
		0.066659,
		0.068984,
		0.052746,
		0.027325,
		0.074616,
		0.054305,
		0.084081,
		0.026501,
		0.043577,
		0.109108,
		0.036466,
		0.030545,
		0.040434,
		0.010651,
		0.030365,
		0.047723,
		0.049605,
		0.028034,
		0.119663,
		0.033854,
		0.145508,
		0.008220,
		0.102297,
		0.035405,
		0.040239,
		0.093627,
		0.097487,
		0.109575,
		0.030081,
		0.046131,
		0.041432,
		0.128666,
		0.089486,
		0.101534,
		0.094124,
		0.028747,
		0.057204,
		0.008693,
		0.018025,
		0.079217,
		0.047818,
		0.040779,
		0.066889,
		0.067823,
		0.052255,
		0.091340,
		-0.000562,
		0.010948,
		-0.015534,
		0.049831,
		0.069419,
		0.105921,
		-0.009779,
		-0.019237,
		-0.000392,
		0.050110,
		0.089361,
		0.040913,
		0.086118,
		0.070317,
		0.062325,
		0.146309,
		0.009333,
		0.033000,
		0.075070,
		0.107140,
		-0.071025,
		0.058276,
		0.069133,
		0.046079,
		0.001198,
		0.051349,
		0.034190,
		0.067158,
		0.097065,
		0.112191,
		0.102745,
		0.137027,
		-0.040221,
		0.112660,
		-0.017135,
		0.047442,
		0.038469,
		0.079301,
		0.213226,
		0.067182,
		0.079999,
		0.022792,
		0.054155,
		0.115451,
		0.025074,
		0.176473,
		0.213415,
		0.012210,
		0.017344,
		0.106191,
		0.037279,
		0.056197,
		0.070956,
		0.087267,
		0.012139,
		-0.018620,
		-0.081607,
		0.010717,
		0.042305,
		0.084384,
		0.029090,
		0.146527,
		0.101202,
		0.064424,
		0.130244,
		0.129149,
		-0.077273,
		0.042417,
		0.067657,
		0.038031,
		0.061063,
		0.009968,
		0.069807,
		0.054760,
		0.075266,
		-0.043706,
		0.093650,
		0.129069,
		0.104751,
		0.034794,
		0.001378,
		0.056335,
		0.028574,
		0.030962,
		0.069805,
		0.133388,
		0.056444,
		0.027387,
		0.045357,
		0.022917,
		0.179956,
		0.099168,
		0.052651,
		0.016702,
		0.018000,
		0.044684,
		0.019092,
		0.099460,
		-0.017390,
		0.031822,
		0.068249,
		0.033872,
		-0.029802,
		0.141002,
		0.042657,
		0.098510,
		0.017476,
		0.093230,
		0.064706,
		0.064115,
		0.047405,
		0.027240,
		0.091330,
		0.082793,
		0.098953,
		0.063581,
		0.081400,
		-0.003682,
		0.069266,
		0.085054,
		0.034980,
		-0.004604,
		0.030342,
		0.089071,
		0.069703,
		0.023755,
		0.094287,
		0.032491,
		-0.023808,
		0.070540,
		0.048103,
		0.088035,
		-0.009201,
		0.029190,
		0.049510,
		0.035616,
		0.043962,
		0.056195,
		0.045582,
		0.101299,
		0.142990,
		0.063979,
		-0.184275,
		0.018743,
		0.094684,
		0.009116,
		0.027808,
		0.112790,
		0.069622,
		-0.073094,
		0.072449,
		-0.008839,
		0.017007,
		0.028071,
		0.001723,
		0.074460,
		0.013415,
		0.067446,
		0.061142,
		-0.024855,
		-0.011063,
		0.021561,
		0.039146,
		0.035110,
		0.097554,
		0.057214,
		0.059547,
		0.047017,
		0.105313,
		0.037903,
		0.092185,
		0.064582,
		0.070365,
		0.130093,
		0.068379,
		0.113301,
		0.064645,
		0.092155,
		0.067475,
		0.009399,
		0.047947,
		0.083165,
		0.071204,
		0.062659,
		0.131234,
		0.049628,
		0.027449,
		0.040134,
		0.075367
	};

	static float prelu_weight[256] = {
		-0.034882,
		0.007089,
		0.021223,
		0.001842,
		-0.165127,
		-0.020082,
		-0.025280,
		-0.057886,
		-0.039741,
		-0.056062,
		-0.079517,
		-0.067483,
		0.079992,
		-0.104331,
		0.008904,
		0.016214,
		-0.159900,
		0.056697,
		-0.015895,
		-0.064988,
		-0.094324,
		-0.060960,
		-0.043386,
		-0.106464,
		0.059612,
		-0.104392,
		-0.082678,
		-0.047170,
		-0.001795,
		-0.083143,
		-0.049372,
		-0.092593,
		-0.002057,
		-0.008074,
		0.012329,
		-0.072265,
		-0.046654,
		-0.013188,
		0.133524,
		0.056477,
		-0.074931,
		-0.019857,
		0.010285,
		0.009508,
		-0.068096,
		-0.085093,
		-0.018832,
		0.011609,
		0.058743,
		-0.056801,
		0.041127,
		-0.101072,
		-0.014827,
		-0.006738,
		-0.064894,
		-0.054550,
		-0.052924,
		-0.028163,
		-0.108636,
		0.085405,
		-0.053090,
		0.007701,
		0.020078,
		-0.018708,
		-0.039806,
		-0.090539,
		-0.168012,
		-0.391911,
		-0.044902,
		0.140780,
		-0.075916,
		-0.128276,
		-0.072540,
		-0.087039,
		0.001241,
		-0.709455,
		0.004240,
		-0.041645,
		0.005976,
		-0.105227,
		0.008817,
		-0.093228,
		-0.067523,
		0.062436,
		-0.051322,
		0.054562,
		0.035675,
		-0.080944,
		0.025013,
		-0.032513,
		-0.477460,
		0.110386,
		-0.045821,
		-0.001102,
		-0.000418,
		-0.005412,
		-0.042278,
		-0.111320,
		-0.042977,
		-0.603858,
		-0.007040,
		-0.019749,
		-0.021454,
		0.024150,
		0.011483,
		-0.043648,
		-0.025790,
		-0.032138,
		-0.061359,
		-0.026159,
		-0.010471,
		0.020197,
		-0.042442,
		-0.196655,
		0.031261,
		0.014138,
		-0.049743,
		-0.005729,
		-0.135030,
		-0.044818,
		0.005436,
		-0.068675,
		-0.636379,
		-0.269047,
		0.012116,
		0.004438,
		-0.001569,
		-0.090323,
		-0.034981,
		-0.056583,
		-0.055015,
		-0.015330,
		-0.032384,
		-0.112723,
		-0.031281,
		-0.727013,
		0.022172,
		-0.020379,
		0.136568,
		-0.078147,
		-0.038227,
		-0.055391,
		-0.433309,
		-0.017445,
		-0.051279,
		-0.008842,
		0.012945,
		0.054079,
		-0.005626,
		-0.016854,
		-0.061482,
		0.009934,
		-0.028641,
		-0.106370,
		-0.023965,
		-0.029638,
		0.107551,
		-0.040674,
		0.057583,
		0.002732,
		-0.015904,
		-0.113511,
		-0.064669,
		-0.027194,
		-0.015832,
		0.161233,
		-0.011962,
		-0.140568,
		-0.034375,
		-0.224808,
		-0.014402,
		0.012009,
		0.006657,
		0.009430,
		0.002328,
		-0.066008,
		-0.111189,
		-0.106258,
		0.014350,
		-0.024752,
		-0.026317,
		-0.074286,
		-0.075056,
		-0.025770,
		0.006329,
		-0.046228,
		-0.042559,
		0.002603,
		0.113826,
		0.047497,
		-0.051161,
		0.010925,
		-0.008650,
		0.000559,
		-0.033642,
		-0.004875,
		-0.067790,
		-0.043995,
		-0.053158,
		0.111315,
		-0.018692,
		-0.066714,
		-0.000668,
		-0.073063,
		-0.043428,
		0.043479,
		-0.067974,
		0.009440,
		-0.044471,
		-0.167656,
		0.050063,
		-0.038946,
		-0.007804,
		-0.063346,
		0.048465,
		-0.011701,
		-0.089973,
		-0.142310,
		0.146011,
		-0.030386,
		-0.049848,
		-0.068994,
		-0.031799,
		0.068940,
		-0.110778,
		-0.037982,
		0.014856,
		0.120645,
		-0.006827,
		-0.012009,
		-0.005419,
		-0.043752,
		-0.012319,
		-0.030135,
		0.001302,
		-0.109335,
		-0.100031,
		-0.002961,
		-0.095768,
		0.001648,
		-0.047705,
		-0.080070,
		0.006936,
		-0.016763,
		0.009731,
		-0.013781,
		0.020271,
		0.023586,
		-0.050715,
		-0.005617,
		-0.030879,
		-0.037819,
		-0.135666,
		0.085571,
		0.018628,
		-0.046355
	};
#pragma HLS ARRAY_PARTITION variable=bias type=complete
#pragma HLS ARRAY_PARTITION variable=prelu_weight type=complete

    float sum = 0;
    int in_offset = 0;
    int weight_offset = 0;
    int data_offset = 0;

	// Convolution layer
    ConvFilter:
	for (int filter = 0; filter < FC1_DENSE_SIZE; filter++) {
#pragma HLS UNROLL off=true
#pragma HLS PIPELINE off=true
		ConvY:
		for (int i = 0; i < CONV4_FILTER * CONV4_OUT_SIZE * CONV4_OUT_SIZE; i++) {
#pragma HLS PIPELINE off=false ii=7
#pragma HLS UNROLL off=true
			weight_offset = filter * CONV4_FILTER * CONV4_OUT_SIZE * CONV4_OUT_SIZE + i;
//			sum = sum + input[i] * weights[weight_offset];
			output[filter] += input[i] * weights[weight_offset];

		}
//		output[filter] = prelu(sum + bias[filter], prelu_weight[filter]);
//		sum = 0;
	}

	for(int i = 0 ; i < FC1_DENSE_SIZE; i++){
		output[i] = prelu(output[i] + bias[i], prelu_weight[i]);
	}
	return;
	
}

void dense_2_1_accel(float* input, /*float* weights,*/ float* output){
// #pragma HLS INTERFACE mode=m_axi port=input offset=slave bundle=gmem0 depth = FC1_DENSE_SIZE 
// #pragma HLS INTERFACE mode=m_axi port=weights offset=slave bundle=gmem1 depth = FC1_DENSE_SIZE * FC2_1_DENSE_SIZE
// #pragma HLS INTERFACE mode=m_axi port=output offset=slave bundle=gmem2 depth = FC2_1_DENSE_SIZE
//#pragma HLS DATAFLOW
	static float bias[2] = {
		-0.174772,
		0.17477
	};
	static float weights[512] = {
		0.029317425563931465, -0.00928312074393034, 0.0434689000248909, 0.2604730427265167, 0.5090525150299072, 0.03001212701201439, 0.07151775807142258, -0.03200992941856384, 0.09690412133932114, 0.05761614814400673, 0.12910348176956177, 0.06857039034366608, 0.010880043730139732, -0.10456148535013199, 0.01656995341181755, 0.00722654489800334, -0.043129466474056244, 0.5590580701828003, -0.02341720275580883, -0.2817162573337555, 0.06180725246667862, 0.01960056461393833, -0.21694301068782806, -0.009167605079710484, 0.036363497376441956, -0.08324519544839859, 0.057394057512283325, 0.04347006976604462, -0.015623747371137142, 0.07171259820461273, 0.034546513110399246, -0.005423611495643854, -0.03313976898789406, 0.01777878776192665, 0.025568775832653046, -0.04047704488039017, -0.09130483120679855, 0.00840983260422945, 0.03403057903051376, 0.501909613609314, 0.14194081723690033, 0.05746084079146385, 0.02959468588232994, -0.01283029280602932, 0.017399862408638, -0.056677669286727905, 0.00905159953981638, 0.028985358774662018, 0.5536206960678101, -0.08440098911523819, 0.05183027684688568, -0.015936514362692833, 0.037023916840553284, 0.008330456912517548, 0.09128204733133316, 0.03947928920388222, 0.09969986975193024, -0.10496177524328232, 0.16027705371379852, 0.14912354946136475, -0.07707647979259491, -0.005970048252493143, -0.005912700667977333, 0.006318651605397463, 0.38428932428359985, 0.01146388053894043, 0.012340724468231201, -0.690132737159729, -0.0049179932102561, 0.017345022410154343, 0.01586964540183544, -0.059554774314165115, -0.17466044425964355, -0.10944666713476181, 0.19376569986343384, -0.9171060919761658, 0.009825577028095722, -0.09781879186630249, 0.00515195494517684, 0.015611283481121063, 0.007579984609037638, 0.024015111848711967, 0.0038318608421832323, 0.008300462737679482, 0.038168761879205704, 0.3978005349636078, 0.4441300928592682, 0.44697970151901245, 0.012869847007095814, 0.11987174302339554, -0.7719704508781433, 0.012027941644191742, 0.0043998658657073975, 0.01840917207300663, 0.02487787790596485, 0.01974913664162159, 0.038490086793899536, 0.12245957553386688, 0.006253237836062908, -0.8810712099075317, 0.022858217358589172, 0.18357644975185394, 0.15311093628406525, 0.1749643087387085, 0.012827753089368343, 0.04696645215153694, 0.034833163022994995, 0.01717107929289341, -0.004326078109443188, -0.021050957962870598, 0.03001057356595993, -0.02363063395023346, 0.03461258113384247, 0.6497534513473511, 0.13335300981998444, 0.0004972763126716018, 0.19750246405601501, 0.021800292655825615, 0.0007636504014953971, 0.027067461982369423, -0.008328499272465706, -0.0645436942577362, -0.9449028968811035, 0.6865348815917969, 0.21814624965190887, 0.011720988899469376, 0.06454496085643768, 0.005657471250742674, 0.3717579245567322, 0.08495747298002243, 0.1240072175860405, 0.02656596153974533, 0.3938172459602356, 0.44609999656677246, 0.12291573733091354, -0.976534903049469, -0.006809561513364315, 0.08367833495140076, 0.025151601061224937, 0.14129963517189026, -0.025492820888757706, 0.04227208346128464, -0.8825302124023438, 0.04212162643671036, 0.026362592354416847, 0.03550098091363907, 0.12077422440052032, 0.06419076770544052, 0.013481318950653076, -0.04924644157290459, -0.144242525100708, 0.22206899523735046, 0.004898148588836193, 0.046582162380218506, -0.04096648097038269, 0.200663223862648, 0.016923420131206512, 0.06345729529857635, 0.5339593291282654, 0.028082100674510002, -0.01003600936383009, -0.3100243806838989, 0.0061133913695812225, 0.12558044493198395, 0.006667491048574448, 0.014971145428717136, 0.10884952545166016, -0.09917955845594406, 0.08494621515274048, -0.31192734837532043, -0.005551645532250404, -0.014543543569743633, 0.04667018726468086, -0.0033786892890930176, 0.09843335300683975, -0.20373216271400452, -0.08940714597702026, -0.05817587673664093, 0.18748272955417633, -0.013549290597438812, 0.12147640436887741, 0.18835148215293884, 0.05559113249182701, 0.018778959289193153, 0.17741046845912933, 0.023138737305998802, 0.02531840093433857, 0.06812890619039536, 0.013629453256726265, 0.0701960027217865, 0.0448625385761261, 0.016989504918456078, -0.06468305736780167, -0.0014224330661818385, 0.06151227653026581, 0.11572031676769257, 0.08830723166465759, 0.023475563153624535, 0.17942851781845093, 0.014076126739382744, 0.011708669364452362, 0.01525200717151165, 0.0028673135675489902, -0.004431953188031912, -0.024691928178071976, 0.09144853800535202, -0.1902856081724167, 0.005270729772746563, 0.0341305248439312, -0.02333524264395237, 0.1716247797012329, 0.04880169779062271, -0.030042696744203568, 0.01673620566725731, 0.641907274723053, 0.00162949925288558, 0.4346184730529785, 0.008749066852033138, 0.008340339176356792, -0.10924402624368668, -0.04192230477929115, -0.02543308585882187, -0.011938204057514668, 0.012401776388287544, -0.46524086594581604, 0.06073848530650139, 0.1671709567308426, 0.010537936352193356, -0.0201045460999012, -0.002820733468979597, 0.017735207453370094, -0.13114111125469208, -0.010940742678940296, 0.043115079402923584, -0.04210295155644417, -0.07724934816360474, -0.0521111935377121, 0.005406091921031475, 0.27901268005371094, 0.013067545369267464, -0.159996896982193, 0.031700022518634796, -0.01977236568927765, 0.05059331655502319, 0.37726473808288574, -0.013843250460922718, -0.0014984734589233994, 0.0049133459106087685, 0.04709293693304062, -0.0017350506968796253, -0.03531040623784065, -0.02672899328172207, -0.29886317253112793, 0.018862301483750343, 0.017578793689608574, -0.05777344852685928, -0.029728621244430542, 0.010811899788677692, -0.042317092418670654, -0.26049497723579407, -0.5080769062042236, -0.02992936410009861, -0.0729127749800682, 0.029513580724596977, -0.09846888482570648, -0.056535057723522186, -0.1276453584432602, -0.06737982481718063, -0.010526539757847786, 0.10211872309446335, -0.015048143453896046, -0.006308031268417835, 0.04152757674455643, -0.5585681200027466, 0.024642108008265495, 0.2821889817714691, -0.06313811242580414, -0.020135128870606422, 0.2167290449142456, 0.009965517558157444, -0.03510446101427078, 0.08368910849094391, -0.05672779679298401, -0.0426221638917923, 0.014958730898797512, -0.0731772929430008, -0.0336812399327755, 0.006216512061655521, 0.03302721679210663, -0.019055336713790894, -0.025897420942783356, 0.04058459401130676, 0.09053149074316025, -0.00968205463141203, -0.0343669168651104, -0.5015623569488525, -0.1405268907546997, -0.059821270406246185, -0.02957690879702568, 0.011685292236506939, -0.01856340654194355, 0.055818166583776474, -0.00924548227339983, -0.029532819986343384, -0.5541728138923645, 0.08492765575647354, -0.0511910505592823, 0.01554106641560793, -0.03764015808701515, -0.00725594162940979, -0.09162279963493347, -0.03973305970430374, -0.09777023643255234, 0.10342037677764893, -0.15911471843719482, -0.15110589563846588, 0.07607601583003998, 0.005527670029550791, 0.004947167821228504, -0.0063379304483532906, -0.3839016258716583, -0.011928130872547626, -0.013405821286141872, 0.6913423538208008, 0.0049551110714674, -0.017366522923111916, -0.017534347251057625, 0.06080184876918793, 0.17719872295856476, 0.1104881763458252, -0.1946469247341156, 0.9156975746154785, -0.010665429756045341, 0.09935061633586884, -0.0050112358294427395, -0.016831254586577415, -0.005938480608165264, -0.02444070763885975, -0.004744995851069689, -0.00740130664780736, -0.03772785887122154, -0.39790207147598267, -0.44440701603889465, -0.4454992115497589, -0.01368450652807951, -0.12052501738071442, 0.7725018262863159, -0.011823819018900394, -0.003282449906691909, -0.019857056438922882, -0.023173673078417778, -0.01866099238395691, -0.03800244629383087, -0.12123377621173859, -0.007852304726839066, 0.8806599974632263, -0.021129705011844635, -0.18267978727817535, -0.15249983966350555, -0.17404542863368988, -0.013532604090869427, -0.04702063277363777, -0.03692825511097908, -0.017032785341143608, 0.004274904262274504, 0.020046042278409004, -0.029487330466508865, 0.02408827841281891, -0.034642137587070465, -0.6499757170677185, -0.13316957652568817, 0.00032108620507642627, -0.1974964588880539, -0.020721182227134705, 0.001151841483078897, -0.02648601122200489, 0.007708813529461622, 0.06616275012493134, 0.946016252040863, -0.6867573261260986, -0.2168869823217392, -0.012300890870392323, -0.06331034749746323, -0.004040491301566362, -0.37196168303489685, -0.08404143899679184, -0.12198109179735184, -0.029082179069519043, -0.39452216029167175, -0.44497811794281006, -0.12243771553039551, 0.976158082485199, 0.006170590873807669, -0.08406458050012589, -0.024584047496318817, -0.14185962080955505, 0.0245176013559103, -0.04180929809808731, 0.8824869990348816, -0.041929710656404495, -0.026081526651978493, -0.03603683412075043, -0.12112002074718475, -0.06441822648048401, -0.012953360565006733, 0.0495791956782341, 0.14510896801948547, -0.2223174273967743, -0.0040634190663695335, -0.04718028008937836, 0.04039064422249794, -0.20230108499526978, -0.01740569807589054, -0.06337130814790726, -0.5347938537597656, -0.029539333656430244, 0.010598712600767612, 0.30857330560684204, -0.005300059914588928, -0.12500278651714325, -0.006844442803412676, -0.014619770459830761, -0.10821307450532913, 0.09891574084758759, -0.08690891414880753, 0.31210857629776, 0.004423379432410002, 0.015604776330292225, -0.047373589128255844, 0.0049401260912418365, -0.09593811631202698, 0.20338810980319977, 0.09008248895406723, 0.05739802122116089, -0.18721376359462738, 0.014535060152411461, -0.12036171555519104, -0.18908697366714478, -0.056403715163469315, -0.019511189311742783, -0.17673037946224213, -0.023370349779725075, -0.02329479530453682, -0.06803702563047409, -0.012354902923107147, -0.07123245298862457, -0.04466995969414711, -0.015426215715706348, 0.06404323130846024, 0.003358270274475217, -0.0628078430891037, -0.11661045253276825, -0.0887894332408905, -0.022656023502349854, -0.1813359409570694, -0.013968565501272678, -0.011550703085958958, -0.014834155328571796, -0.0018462407169863582, 0.005201086867600679, 0.02465687319636345, -0.08967199921607971, 0.18855834007263184, -0.003471645060926676, -0.03292132914066315, 0.02437753789126873, -0.17283660173416138, -0.04762553796172142, 0.03042520396411419, -0.01867058500647545, -0.6416109800338745, -0.0018671057187020779, -0.43398216366767883, -0.008395539596676826, -0.008667177520692348, 0.10947643965482712, 0.0437135323882103, 0.024390459060668945, 0.012505297549068928, -0.01104829739779234, 0.46589013934135437, -0.05896247178316116, -0.16619142889976501, -0.01004963368177414, 0.021373573690652847, 0.002761540003120899, -0.019994383677840233, 0.1318562626838684, 0.010776348412036896, -0.04244412109255791, 0.04181879386305809, 0.07676862925291061, 0.05358777567744255, -0.006492104381322861, -0.2775176167488098, -0.012639510445296764, 0.16072259843349457, -0.032365646213293076, 0.020394466817378998, -0.0520734079182148, -0.3772854506969452, 0.01323945727199316, 0.00026188057381659746, -0.005322638899087906, -0.046502452343702316, 0.0024647850077599287, 0.03476962819695473, 0.02523324266076088, 0.2983202040195465, -0.019056150689721107, -0.01902191713452339, 0.05623401701450348
	};
#pragma HLS ARRAY_PARTITION variable=bias type=complete
// #pragma HLS ARRAY_PARTITION variable=prelu_weight type=complete

    float sum = 0;
    int in_offset = 0;
    int weight_offset = 0;
    int data_offset = 0;

	// Convolution layer
    ConvFilter:
	for (int filter = 0; filter < FC2_1_DENSE_SIZE; filter++) {
//#pragma HLS UNROLL off=true
//#pragma HLS PIPELINE off=true
		ConvY:
		for (int i = 0; i < FC1_DENSE_SIZE ;i++) {
#pragma HLS PIPELINE off=false ii=7
//#pragma HLS UNROLL off=true
			weight_offset = filter * FC1_DENSE_SIZE + i;
			sum = sum + input[i] * weights[weight_offset];
			output[filter] += input[i] * weights[weight_offset];

		}
//		output[filter] = sum + bias[filter];
//		sum = 0;
	}

	for(int i = 0 ; i < FC2_1_DENSE_SIZE; i++){
		output[i] += bias[i];
	}

	return;
	
}

void dense_2_2_accel(float* input, /*float* weights,*/ float* output){
// #pragma HLS INTERFACE mode=m_axi port=input offset=slave bundle=gmem0 depth = FC1_DENSE_SIZE 
// #pragma HLS INTERFACE mode=m_axi port=weights offset=slave bundle=gmem1 depth = FC1_DENSE_SIZE * FC2_2_DENSE_SIZE
// #pragma HLS INTERFACE mode=m_axi port=output offset=slave bundle=gmem2 depth = FC2_2_DENSE_SIZE
//#pragma HLS DATAFLOW
	static float bias[4] = {
		0.099938,
		0.027741,
		-0.099611,
		-0.009517
	};

	static float weights[1024] = {
		0.04369400069117546, -0.04591478779911995, -0.15634331107139587, 0.0024122903123497963, -0.0016747904010117054, -0.004975839052349329, -0.004912071395665407, 0.050522442907094955, -0.1283007264137268, 0.004186955746263266, 0.054500680416822433, -0.12497305124998093, -0.00786372646689415, 0.04195752367377281, -0.01147034578025341, -0.009825226850807667, -0.0009277441422455013, 0.030855679884552956, -0.10406879335641861, -0.021617313846945763, -0.01827375777065754, 0.013972245156764984, 0.055176131427288055, -0.09455570578575134, -0.018106097355484962, -0.02189025469124317, 0.04711257293820381, 0.021508246660232544, 0.00045357708586379886, -0.07544966042041779, -0.0416979044675827, 0.27994289994239807, -0.035257741808891296, 0.002843855880200863, -0.003772585652768612, -0.07353845983743668, -0.06068888679146767, -0.020917562767863274, -0.016919130459427834, 0.002738099079579115, -0.015821613371372223, -0.009454584680497646, -0.06852662563323975, 0.015924761071801186, -0.083346426486969, -0.15295298397541046, -0.02964426949620247, -0.02110808901488781, -0.0480814166367054, -0.018784528598189354, 0.014022343792021275, -0.2656404674053192, -0.011346654035151005, -0.00019978125055786222, 0.14952941238880157, -0.00904013030230999, -0.21288147568702698, -0.08420170843601227, 0.04434245079755783, -0.07208217680454254, -0.07665403187274933, 0.02235257625579834, -0.037708427757024765, -0.024166438728570938, -0.011049128137528896, -0.022087518125772476, 0.08935943245887756, -0.012181170284748077, -0.03194063901901245, -0.0009583396022208035, 0.05999540910124779, -0.08430531620979309, -0.09742943942546844, -0.007626398466527462, -0.030611351132392883, 0.006905939895659685, 0.009915383532643318, 0.07201965898275375, -0.008558235131204128, 0.02973863296210766, -0.0017202731687575579, 0.11178357899188995, -0.0204592514783144, -0.02319897711277008, -0.2038988322019577, 0.014346909709274769, -0.0034470614045858383, -0.007512784097343683, -0.0054171825759112835, 0.0744270533323288, -0.010503822937607765, 0.001734954654239118, 0.14333505928516388, 0.00815706979483366, 0.023612691089510918, 0.014154662378132343, 0.014171164482831955, 0.2776396870613098, 0.013563238084316254, -0.015757402405142784, -0.004326558671891689, -0.018937984481453896, -0.06305290013551712, -0.09274739772081375, -0.017405560240149498, -0.011284900829195976, 0.04687681421637535, -0.016107870265841484, 0.18920566141605377, -0.06483004987239838, 0.057734064757823944, -0.032845743000507355, 0.06771805137395859, 0.01692287065088749, -0.011611477471888065, 0.042039912194013596, 0.04670645669102669, -0.02044348604977131, 0.30529913306236267, 0.015186970122158527, -0.006144723389297724, -0.16444993019104004, -0.005970703903585672, -0.010461879894137383, 0.09326039254665375, -0.01164277084171772, 0.005080080591142178, 0.13286827504634857, 0.014208940789103508, -0.030226338654756546, 0.04194778949022293, 0.010148587636649609, -0.033518578857183456, 0.030857238918542862, -0.13057208061218262, 0.0026974366046488285, -0.012647243216633797, 0.13484357297420502, -0.06659042835235596, -0.047743748873472214, -0.007275748997926712, 0.03361712396144867, 0.001681973459199071, 0.019087722525000572, 0.033198174089193344, -0.0178796648979187, -0.10718991607427597, -0.04019464924931526, 0.051988840103149414, -0.025468576699495316, 0.013099652715027332, 0.003174291690811515, 4.938468191539869e-05, -0.05436887592077255, -0.13493895530700684, -0.012218933552503586, -0.03407777100801468, 0.039583634585142136, -0.04898655787110329, 0.08277083188295364, 0.016083072870969772, 0.04826389625668526, 0.1107662245631218, 0.002689545741304755, 0.02442855015397072, -0.023000789806246758, 0.07313378155231476, 0.06830460578203201, -0.15726111829280853, 0.1526746153831482, -0.021839546039700508, 0.04982549324631691, -0.01479396503418684, -0.026549430564045906, 0.043359629809856415, -0.0032115827780216932, 0.044916268438100815, -0.0847056657075882, 0.01494769286364317, -0.04313439875841141, 0.014285854063928127, 0.07756441086530685, -0.03484199196100235, -0.0013152197934687138, 0.021500734612345695, 0.03862694278359413, -0.025083502754569054, -0.11324626207351685, -0.0015471276128664613, -0.007793730590492487, -0.014294709078967571, -0.004152569454163313, -0.10796646773815155, 0.06792233884334564, -0.027045495808124542, 0.016687382012605667, -0.023687951266765594, -0.01760423555970192, 0.0018865375313907862, -0.007442255970090628, 0.07174161076545715, -0.09534967690706253, -0.048192210495471954, -0.18143907189369202, -0.003980858251452446, -0.058667704463005066, -0.0011827705893665552, 0.03986160829663277, -0.011509036645293236, 0.07649990916252136, 0.022295156493782997, -0.018412334844470024, -0.008008884266018867, -0.09734180569648743, 0.034933894872665405, -0.03777216374874115, 0.00669279508292675, -0.06243261322379112, -0.01419305894523859, 0.007232106290757656, -0.11027936637401581, 0.07273439317941666, 0.08764167129993439, -0.012837203219532967, 0.012777338735759258, -0.029429960995912552, 0.0263883825391531, 0.009683987125754356, -0.06031465530395508, -0.025019297376275063, -0.027210572734475136, -0.010787051171064377, -0.051228635013103485, 0.030699262395501137, 0.02550489827990532, -0.037384774535894394, 0.09339287132024765, 0.0010519006755203009, -0.040711820125579834, -0.012313754297792912, 0.026392115280032158, -0.01165349967777729, -0.12012254446744919, -0.05198485031723976, -0.023279987275600433, -0.04667207971215248, -0.013657032512128353, 0.03300429880619049, 0.009644792415201664, 0.03848874568939209, -0.04339216277003288, 0.007994246669113636, -0.18859924376010895, 0.011509987525641918, -0.023220058530569077, -0.008577031083405018, 0.017910072579979897, -0.0023856069892644882, 0.07382214069366455, -0.01386366318911314, -0.04388470947742462, 0.013588441535830498, 0.026082834228873253, 0.23138602077960968, 0.009875050745904446, 0.033936530351638794, -0.062421370297670364, -0.12493050843477249, 0.005124708637595177, -0.005353621672838926, 0.011521784588694572, 0.009455885738134384, 0.05824728310108185, 0.023951832205057144, 0.05005159601569176, 0.14197880029678345, -0.008890626952052116, 0.06708534061908722, -0.0013195399660617113, -0.25847163796424866, -0.04045398533344269, -0.010699287056922913, -0.09680547565221786, 0.008826724253594875, -0.038154009729623795, -0.08506640046834946, -0.010847360827028751, 0.12115564197301865, 0.044079236686229706, -0.02782154455780983, 0.04049177095293999, -0.039666254073381424, -0.009720017202198505, 0.005355666391551495, -0.0032631107605993748, 0.012271085754036903, -0.04875996708869934, -0.0158607829362154, -0.05726271867752075, 0.008700612932443619, -0.026619821786880493, -0.014095531776547432, -0.033262502402067184, 0.01539265364408493, -0.02395911142230034, 0.03201054781675339, 0.01581183448433876, -0.035459768027067184, 0.01340221893042326, 0.010079787112772465, -0.038455650210380554, -0.003766506677493453, -0.03295912966132164, 0.016638146713376045, -0.009158489294350147, -0.011805101297795773, -0.0510534904897213, -0.005302992649376392, 0.027072304859757423, 0.02950011007487774, -0.01112064067274332, -0.24840688705444336, -0.017321674153208733, -0.010090473107993603, -0.11671119183301926, -0.011079496704041958, 0.07342808693647385, -0.2614516019821167, 0.032407376915216446, -0.02814105711877346, -0.027886880561709404, 0.006083795800805092, 0.05661864951252937, 0.020108662545681, -0.008189184591174126, 0.28005683422088623, -0.027459898963570595, 0.24105961620807648, 0.0002855211205314845, -0.08157284557819366, -0.12789922952651978, -0.04731404781341553, -0.021995535120368004, -0.053859904408454895, 0.013525809161365032, -0.04550713300704956, 0.0010536561021581292, -0.018081407994031906, 0.018862172961235046, -0.017002586275339127, -0.029610594734549522, 0.026796886697411537, -0.019574863836169243, 0.026600195094943047, 0.11587408930063248, -0.00728607689961791, -0.0027481946162879467, 0.05065328627824783, -0.04859893023967743, -0.009752044454216957, 0.000682117766700685, -0.0011254905257374048, 0.1302943378686905, -0.04816875979304314, 0.03883962333202362, -0.06653884798288345, 0.020974472165107727, -0.12344101071357727, -0.0047466568648815155, 0.0014191813534125686, -0.13578464090824127, -0.1142292395234108, 0.08651583641767502, -0.020629307255148888, 0.06780334562063217, 0.0006804214208386838, -0.0010142409009858966, -0.09493137151002884, -0.0013627030421048403, -6.781762931495905e-05, 0.030776863917708397, -0.02077636867761612, 0.012318793684244156, -0.15626265108585358, 0.09578591585159302, -0.042088016867637634, 0.02108835056424141, -0.014673099853098392, 0.007683962117880583, 0.04014696180820465, -0.15743455290794373, -6.320128159131855e-05, 0.0009094863198697567, 0.10377262532711029, -0.08209969848394394, -0.012490716762840748, -0.05220356583595276, 0.0021491143852472305, -0.0036042025312781334, -0.024906955659389496, -0.0008344504749402404, -0.01770242489874363, -0.16951002180576324, -0.009401974268257618, -0.020416824147105217, -0.01241164468228817, -0.05255243927240372, 0.025095036253333092, 0.02576279640197754, -0.4468493163585663, -0.05530736222863197, -0.06543051451444626, 0.006527981720864773, -0.006284249015152454, -0.002126123756170273, 0.08939856290817261, 0.06622608751058578, 0.03534804284572601, 0.14776861667633057, 0.1453447788953781, 0.010277815163135529, 0.020643791183829308, 0.09758789092302322, 0.2632140517234802, -0.055095892399549484, 0.1318483054637909, -0.011267948895692825, 0.0017872812459245324, -0.0462176688015461, 0.007604677230119705, -0.03139824420213699, -0.14401084184646606, 0.15814147889614105, -0.010480438359081745, -0.09331482648849487, -0.012691188603639603, -0.000377924443455413, 0.12219066172838211, 0.16623593866825104, 0.013532034121453762, -0.08922936767339706, 0.010025452822446823, -0.00269074784591794, 0.047171700745821, 0.0006283880211412907, -0.005022742785513401, -0.00642125029116869, 0.00925426185131073, -0.06328458338975906, -0.02631748467683792, -0.038507718592882156, 0.04968014359474182, 0.002829953795298934, 0.008801409043371677, -0.025637848302721977, -0.0006430443609133363, -0.1288887858390808, -0.2164004147052765, 0.00900453981012106, -0.07911339402198792, 0.002943339990451932, -0.1040867269039154, -0.02011413313448429, -0.003276038682088256, -0.00011384099343558773, 0.038787052035331726, 0.05109114199876785, 0.005194223485887051, 0.06913866102695465, -0.035388000309467316, 0.014444578438997269, -0.007754133082926273, 0.05075956508517265, 0.09478874504566193, 0.010587315075099468, -0.21263594925403595, -0.05822521448135376, -0.11517379432916641, 0.023286165669560432, -0.011375966481864452, 0.012877708300948143, -0.0034073868300765753, -0.08087144792079926, -0.008320948109030724, -0.11488901823759079, -0.010957641527056694, -0.016851702705025673, -0.005372459068894386, -0.015328728593885899, -0.05116238817572594, -0.018763016909360886, -0.0014694016426801682, -0.07090160250663757, -0.021129170432686806, -0.18404227495193481, 0.0038143545389175415, -0.021941792219877243, -0.003513404168188572, -0.03252706676721573, 0.007865066640079021, 0.010280563496053219, -0.027688538655638695, 0.014617379754781723, 0.015799768269062042, 0.016630128026008606, 0.008197042159736156, -0.08663441985845566, 0.0761290192604065, -0.09133677929639816, -0.03834307938814163, 0.053507447242736816, -0.10604985803365707, -0.003722919151186943, 0.022792385891079903, 0.05527624487876892, 0.02397748827934265, 0.0012784129939973354, -0.016044219955801964, -0.009920970536768436, -0.049557846039533615, 0.014903428964316845, -0.12787246704101562, 0.0002885025169234723, 0.08601347357034683, 0.012469072826206684, -0.034109607338905334, 0.0629405528306961, -0.013213316909968853, -0.03931829333305359, 0.013572828844189644, -0.004536111373454332, -0.062426745891571045, 0.015622773207724094, 0.036399636417627335, -0.04961872845888138, 0.0030791654717177153, -0.1307496428489685, 0.08055678755044937, 0.269614577293396, -0.006252954714000225, 0.02208809368312359, 0.12116985768079758, -0.0167803093791008, 0.057044003158807755, -0.0952552929520607, 0.007893393747508526, 0.08373011648654938, 0.10680173337459564, 0.02060706727206707, 0.021502699702978134, 0.026825888082385063, -0.0017892020987346768, 0.20053130388259888, -0.008937371894717216, 0.08543407917022705, 0.01266605406999588, 0.08492345362901688, -0.05360189080238342, 0.018703298643231392, 0.003244640538468957, 0.022205503657460213, 0.05472951382398605, 0.0020417210180312395, -0.09901487082242966, 0.025381961837410927, 0.011363494209945202, -0.00785659346729517, -0.020534032955765724, -0.036927301436662674, -0.056759193539619446, 0.23547106981277466, -0.13273106515407562, 0.016631213948130608, -0.0041110822930932045, 0.09117119014263153, -0.0777086392045021, 0.009526134468615055, 0.14490677416324615, -0.07915636897087097, 0.015630148351192474, 0.1415286511182785, 0.012965968810021877, -0.13462670147418976, 0.08490844815969467, -0.016620568931102753, -0.13233257830142975, -0.016360344365239143, 0.007680618204176426, 0.049265578389167786, 0.08484239131212234, 0.009425266645848751, -0.10090480744838715, -0.037237219512462616, -0.13557617366313934, -0.005116171669214964, -0.045113880187273026, 0.04020560905337334, -0.014227781444787979, 0.035508137196302414, -0.017456727102398872, 0.02091868221759796, 0.011203693225979805, -0.0049989065155386925, 0.00476333312690258, 0.1640585958957672, 0.01569434255361557, 0.03300553187727928, -0.0007474805461242795, 0.013253506273031235, 0.09391680359840393, -0.006836336571723223, -0.008103466592729092, -0.013863224536180496, -0.06093747913837433, 0.11867596954107285, 0.009763026610016823, 0.003641961608082056, -0.026937834918498993, -0.029128801077604294, 0.0762631967663765, 0.1421479731798172, 0.10422609746456146, 0.07720956951379776, 0.041150275617837906, 0.02657262608408928, 0.007771733682602644, 0.07020708918571472, -0.06598329544067383, -0.039481375366449356, 0.09870345145463943, 0.06456933170557022, -0.003752799704670906, 0.006898890249431133, -0.12803691625595093, 7.703276060055941e-05, -0.005158398766070604, 0.05155254900455475, -0.020382409915328026, 0.0022206443827599287, 0.19599920511245728, 0.004749147221446037, -0.020230021327733994, -0.021682975813746452, -0.008722871541976929, -0.061759188771247864, -0.008033731020987034, 0.08779694885015488, 0.0010638763196766376, 0.0038336541038006544, 0.026254432275891304, -0.0025042423512786627, -0.1836138516664505, 0.0867389589548111, -0.007367619313299656, 0.006109421607106924, 0.02184976451098919, 0.008580422960221767, -0.013177504763007164, 0.08203832060098648, -0.028634237125515938, 0.03612929955124855, -0.1902514547109604, 0.008558464236557484, -0.0027798113878816366, -0.019794048741459846, 0.09320185333490372, 0.01826508343219757, 0.06413928419351578, 0.0318521186709404, 0.037945374846458435, -0.05140145868062973, -0.02804027497768402, 0.1257287859916687, -0.05649951845407486, -0.0634007602930069, -0.05004405230283737, -0.04747510328888893, 0.022663623094558716, 0.0012001801514998078, 0.015863241627812386, -0.05767011642456055, -0.04167778789997101, -0.012956763617694378, -0.012829295359551907, -0.016295209527015686, -0.006182260811328888, -0.03796868398785591, 0.04279174283146858, -0.14407707750797272, 0.07207084447145462, 0.09920252114534378, -0.026870183646678925, -0.02183065377175808, -0.12881481647491455, 0.02943740226328373, -0.01230480708181858, 0.029668882489204407, 0.11677878350019455, 0.02742006629705429, -0.052667997777462006, 0.013435179367661476, 0.09072785079479218, -0.03593788295984268, 0.010822009295225143, -0.034983955323696136, -0.05912409722805023, 0.15013107657432556, -0.08481059968471527, -0.007637627422809601, -0.004332637879997492, 0.03981661796569824, 0.011231047101318836, 0.10803680866956711, 0.01548547949641943, 0.04704293608665466, -0.040149156004190445, 0.03485449403524399, -0.0625549927353859, 0.024937715381383896, -0.002810648176819086, 0.018416069447994232, 0.02965579181909561, 0.05405376851558685, -0.02729068696498871, 0.057065416127443314, 0.0792563185095787, 0.02514275722205639, -0.008189097978174686, 0.026445599272847176, -0.34438008069992065, 0.016511188820004463, 0.009824954904615879, -0.15528914332389832, 0.18461398780345917, -0.013958416879177094, 0.00151862355414778, 0.07040797173976898, 0.009085934609174728, 0.038892656564712524, -0.0019063116051256657, 0.1028289943933487, -0.07235732674598694, 0.014736324548721313, -0.007255339529365301, 0.001532471738755703, 0.2263135462999344, 0.009878681041300297, -0.16927847266197205, 0.1334173083305359, 0.06660837680101395, 0.08230867236852646, -0.011330231092870235, 0.025700371712446213, 0.003419396933168173, 0.0030096201226115227, 0.012121636420488358, -0.03126499429345131, -0.0496736541390419, -0.0029635545797646046, 0.04315908998250961, -0.004578720778226852, 0.02130432426929474, 0.0589071586728096, -0.04425536096096039, -0.03521614149212837, 0.0019319719867780805, -0.02371748350560665, 0.04033135250210762, -0.03141803666949272, -0.003341298783197999, 0.025172878056764603, 0.04721396043896675, 0.0014132545329630375, 0.012806194834411144, -0.014374480582773685, 0.15838530659675598, 0.19965550303459167, -0.12829172611236572, -0.05944136530160904, 0.1868460327386856, 0.008379138074815273, -0.21579666435718536, -0.027978084981441498, 0.015165049582719803, -0.02053188718855381, -0.0031285914592444897, 0.0363902673125267, 0.07331836223602295, -0.020596418529748917, 0.0734303817152977, -0.048873260617256165, -0.12011214345693588, -0.08738734573125839, 0.24990035593509674, 0.10201218724250793, -0.0011335393646731973, 0.06631365418434143, 0.1976000815629959, 0.016398707404732704, -0.06407498568296432, -0.039198990911245346, 0.045142821967601776, 0.11919163912534714, 0.24215053021907806, 0.14312979578971863, 0.013839710503816605, -0.022660119459033012, -0.043736062943935394, 0.22937782108783722, 0.004781415686011314, 0.12949314713478088, 0.008358647115528584, 0.20982061326503754, 0.11712072044610977, 0.000526498886756599, 0.026329083368182182, 0.023587854579091072, -0.039882585406303406, -0.015532629564404488, 0.06963635236024857, -0.007128018885850906, 0.0042837620712816715, -0.2257845550775528, 0.01078244112432003, 0.1683558076620102, 0.0468648262321949, 0.16259443759918213, -0.033393651247024536, 0.05421539023518562, -0.0038922904059290886, -0.045359645038843155, 0.08821269869804382, 0.0020965919829905033, -0.03783944994211197, -0.36110275983810425, 0.003187945345416665, 0.08269591629505157, -0.012884593568742275, -0.21521760523319244, 0.017122261226177216, 0.04752582684159279, -0.19294264912605286, 0.016361581161618233, 0.001324332319200039, -0.0015682803932577372, 0.016450779512524605, -0.0017374929739162326, -0.01823817752301693, 0.02438604086637497, -0.06222459673881531, 0.007623317185789347, 0.010982795618474483, 0.06268969178199768, -0.008532779291272163, 0.008089357055723667, -0.00374361639842391, 0.0030466539319604635, -0.061788950115442276, 0.005416386295109987, -0.007410385180264711, 0.07966198772192001, 0.037904560565948486, 0.010941315442323685, -0.02015495114028454, -0.0020602161530405283, -0.025079550221562386, 0.04724164679646492, 0.007544863969087601, 0.020691975951194763, 0.03553345054388046, 0.07579082995653152, 0.0038589441683143377, -0.0005888411542400718, 0.021996157243847847, 0.015237327665090561, -0.022400151938199997, 0.008923720568418503, 0.08386368304491043, -0.007511545438319445, 0.036016855388879776, -0.015099121257662773, -0.011054066941142082, -0.023665331304073334, 0.184965118765831, -0.07247062027454376, -0.06565380096435547, -0.13456347584724426, -0.022573214024305344, 0.0055677685886621475, 0.04342756047844887, -0.013657856732606888, 0.009187119081616402, 0.002402722369879484, -0.022734642028808594, -0.019260989502072334, 0.1675509661436081, 0.03202741593122482, -0.04791167005896568, -0.014094468206167221, -0.007707341108471155, -0.034202396869659424, -0.009307920932769775, 0.04952678084373474, -0.007605878636240959, -0.003663969226181507, 0.015634246170520782, -0.017437811940908432, -0.138225719332695, -0.017879927530884743, -0.10149436444044113, -0.004853392019867897, 0.0028628073632717133, -0.004680200945585966, 0.024239936843514442, 0.057547785341739655, -0.029502803459763527, -0.017080020159482956, 0.009386536665260792, 0.0695972815155983, 0.012418339028954506, -0.002582743763923645, -0.09485678374767303, -0.07129991054534912, 0.08873752504587173, 0.013368957675993443, -0.016421666368842125, 0.02304091304540634, -0.032391250133514404, 0.08111949265003204, -0.0695362538099289, 0.01416323147714138, 0.09634017944335938, 0.001987519208341837, 0.019585082307457924, -0.09602732211351395, 0.1804388463497162, 0.06446030735969543, -0.07084439694881439, 0.0022340731229633093, 0.013096551410853863, 0.008597838692367077, 0.0065054502338171005, 0.010174787603318691, -0.022162700071930885, -0.04144209250807762, 0.2880476415157318, 0.03990403935313225, 0.023170510306954384, -0.00818333774805069, -0.13079385459423065, 0.44227689504623413, -0.007452476304024458, -0.07162472605705261, 0.01848033256828785, 0.03597082570195198, 0.1508604735136032, 0.015406155027449131, 0.04375344142317772, -0.03400807082653046, -0.01160476915538311, 0.026564819738268852, 0.007559461053460836, 0.22559195756912231, -0.12190733104944229, 0.022559667006134987, 0.03175142779946327, 0.06478061527013779, -0.008214212022721767, 0.07675617933273315, 0.04914192110300064, -0.014837385155260563, 0.15868720412254333, -0.007953987456858158, -0.01867028884589672, 0.036975957453250885, 0.010139879770576954, 0.015369478613138199, -0.0398561954498291, 0.009099811315536499, 0.03174649924039841, -0.05376826971769333, 0.20702984929084778, 0.03478873521089554, -0.01208004541695118, 0.01976615935564041, -0.08046310395002365, -0.0025296967942267656, 0.02812415361404419, 0.1440870314836502, 0.05816977098584175, -0.031089862808585167, 0.01991751790046692, -0.02282380685210228, 0.00822595413774252, -0.060256555676460266, 0.0175944734364748, 0.10174639523029327, -0.01776965521275997, 0.0013075580354779959, 0.033752329647541046, -0.04288802668452263, -0.007833940908312798, 0.047563593834638596, -0.12830229103565216, -0.006351431831717491, -0.03495892509818077, 0.07105153799057007, 0.00277742394246161, -0.00936433020979166, 0.01245675329118967, -0.08512698858976364, -0.02729281783103943, -0.03625783324241638, -0.09804590046405792, -0.015056448057293892, -0.013378346338868141, 0.10181485861539841, -0.008394723758101463, 0.1094789057970047, 0.04322522506117821, 0.003914548084139824, 0.022200407460331917, 0.034078001976013184, -0.04382118955254555
	};
#pragma HLS ARRAY_PARTITION variable=bias type=complete
#pragma HLS BIND_STORAGE variable=weights type=ram_s2p impl=lutram

    float sum = 0;
    int in_offset = 0;
    int weight_offset = 0;
    int data_offset = 0;

	// Convolution layer
    ConvFilter:
	for (int filter = 0; filter < FC2_2_DENSE_SIZE; filter++) {
//#pragma HLS UNROLL off=true
//#pragma HLS PIPELINE off=true
		ConvY:
		for (int i = 0; i < FC1_DENSE_SIZE ;i++) {
#pragma HLS PIPELINE off=false ii=7
//#pragma HLS UNROLL off=true
			weight_offset = filter * FC1_DENSE_SIZE + i;
			sum = sum + input[i] * weights[weight_offset];
			output[filter] += input[i] * weights[weight_offset];

		}
//		output[filter] = sum + bias[filter];
//		sum = 0;
	}

	for(int i = 0 ; i < FC2_2_DENSE_SIZE; i++){
		output[i] += bias[i];
	}

	return;
	
}
void dense_2_3_accel(float* input, float* output){
// #pragma HLS INTERFACE mode=m_axi port=input offset=slave bundle=gmem0 depth = FC1_DENSE_SIZE 
// #pragma HLS INTERFACE mode=m_axi port=weights offset=slave bundle=gmem1 depth = FC1_DENSE_SIZE * FC2_3_DENSE_SIZE
// #pragma HLS INTERFACE mode=m_axi port=output offset=slave bundle=gmem2 depth = FC2_3_DENSE_SIZE
//#pragma HLS DATAFLOW
	static float bias[10] = {
		0.339874,
		0.688847,
		0.517248,
		0.365005,
		0.661452,
		0.381809,
		0.375047,
		0.587197,
		0.754140,
		0.746857
	};
	static float weights[2560] = {
		0.020405491814017296, -0.06263376772403717, 0.038624107837677, 0.04946097359061241, -0.006044425070285797, -0.03822747990489006, 0.03035474196076393, 0.010085665620863438, -0.023687371984124184, 0.039387039840221405, 0.08001307398080826, -0.024763643741607666, 0.020282795652747154, 0.0058735026977956295, -0.028080519288778305, -0.0378027968108654, -0.16520164906978607, 0.012370169162750244, -0.010421116836369038, 0.00017025021952576935, 0.004047010093927383, -0.002444916870445013, 0.010463396087288857, -0.08022932708263397, 0.005737623665481806, -0.005483746528625488, 0.003257635049521923, 0.03065439499914646, -0.0026867056731134653, -0.004220847971737385, -0.0009375730878673494, 0.05548455938696861, 8.89146322151646e-05, -0.015908388420939445, -0.005022360011935234, -0.021503260359168053, 0.05039028450846672, -0.025859225541353226, 0.018962910398840904, 0.06230100616812706, -0.02062477171421051, 0.04832759127020836, -0.03374030441045761, -0.0005844851839356124, -0.011046972125768661, -0.027434444054961205, -0.0031717410311102867, -0.000766855722758919, -0.019581371918320656, 0.119334377348423, -0.02236911468207836, -0.08391785621643066, 0.003320498624816537, -0.0001541500969324261, 0.05383042246103287, -0.00038492571911774576, -0.01586971990764141, -0.004446974955499172, 0.014968954026699066, -0.022460300475358963, -0.03917182609438896, -0.0028310713823884726, 0.037444643676280975, 0.004801293835043907, -0.004861282650381327, 0.062005627900362015, 0.0017312223790213466, -0.00024355547793675214, -0.014394485391676426, -0.04898807406425476, -0.003290367079898715, -0.02598041109740734, -0.058335840702056885, 0.03127417340874672, 0.0029865186661481857, 0.00011048941087210551, 0.021542631089687347, 0.11406605690717697, -0.02901320718228817, 0.0033366610296070576, -0.11685018241405487, 0.010934138670563698, -0.12106039375066757, 0.04757554456591606, -0.039250146597623825, -0.002971642417833209, -0.003948644269257784, 0.003163093002513051, -0.0004916390753351152, 0.07794167846441269, -0.0008962977444753051, 0.005532902665436268, 0.020083004608750343, 0.0015420791460201144, 0.006567042320966721, 0.010994648560881615, -0.0307964775711298, 0.03199632838368416, 0.011057482101023197, -0.0006747774896211922, 0.0305474940687418, 0.0026137586683034897, -0.009484102949500084, -0.03974870964884758, 0.010527552105486393, 0.05792119726538658, 0.017389649525284767, -0.005388010758906603, 0.03274912387132645, 0.0297805517911911, 0.02256198413670063, -0.0012085747439414263, 0.023524634540081024, -0.0008031547185964882, 0.00584053248167038, 0.013185243122279644, 0.021807806566357613, 0.005893079098314047, 0.014549529179930687, 0.0028047431260347366, -0.0004475215682759881, 0.02595604956150055, -0.0006207161350175738, -0.0015358706004917622, 0.02682780660688877, 0.001026918995194137, 0.04154141619801521, 0.043520502746105194, 0.024951504543423653, -0.033387914299964905, -0.03309798985719681, -0.004055018536746502, -0.02781154215335846, 0.0009070344967767596, -0.041205570101737976, -0.0003441880689933896, -0.010603154078125954, 0.0519777350127697, -0.05361296609044075, -0.01798427291214466, -0.037719886749982834, -0.014968572184443474, -0.0007900053751654923, -0.02092435210943222, 0.17559021711349487, -0.015363222919404507, -0.02421391010284424, -0.045668382197618484, 0.016843020915985107, -0.0005344885867089033, -0.015597978606820107, 0.005537824705243111, 0.00376319931820035, -0.021648088470101357, 0.021326465532183647, -0.019400106742978096, -0.003343593794852495, 0.05027446523308754, 0.021717002615332603, 0.012837174348533154, -0.019551396369934082, 0.012072083540260792, 0.0919293686747551, 0.007992849685251713, -0.013165767304599285, 0.005787277594208717, 0.006888081319630146, -0.015613560564815998, -0.0722116231918335, 0.0095058623701334, -0.0036522422451525927, 0.011021871119737625, 0.005689735524356365, -0.06085391342639923, -0.012664183974266052, -0.0012484922772273421, -0.04053102061152458, -0.001471076044254005, 0.035165492445230484, 0.01566353254020214, -0.04349955916404724, -0.0018020231509581208, 0.010844018310308456, -0.009988623671233654, 0.007341587450355291, 0.013702250085771084, 0.0018685200484469533, -0.035022489726543427, -0.020030593499541283, -0.024423694238066673, -0.04835373908281326, -0.006399226374924183, 0.0040282923728227615, -0.003216535784304142, -0.015867339447140694, 0.005683240480720997, -0.12393953651189804, 0.0017440030351281166, -0.015274317003786564, -0.014415817335247993, 0.08233802765607834, -0.021972572430968285, -0.0014670176897197962, -0.009225834161043167, 0.00298596010543406, 0.03027593530714512, -0.025183187797665596, 0.1150527223944664, -0.06788681447505951, 0.14644452929496765, 0.040089335292577744, 0.008685985580086708, -0.05207164213061333, -0.04511800408363342, -0.03013482503592968, -0.012681965716183186, -0.0040055448189377785, -0.0006911198724992573, 0.0244672242552042, 0.0003966620715800673, 0.03200570493936539, 0.045564793050289154, 0.06741759926080704, 0.13321608304977417, -0.0012837373651564121, -0.1526552140712738, -0.008079653605818748, -0.02864731103181839, 0.002110726200044155, 0.030938230454921722, -0.0309648048132658, -0.009048672392964363, -0.07830484956502914, 0.008964139968156815, -0.0012196967145428061, -0.008864718489348888, 0.06587269902229309, -0.03707234188914299, -0.004222240764647722, -0.005487073212862015, 0.01982392929494381, 0.20412679016590118, 0.015062803402543068, 0.04056220501661301, -0.05645338445901871, 0.0034686170984059572, -0.05026094987988472, 0.012021251022815704, -0.01355336606502533, 0.010324297472834587, -0.035402730107307434, 0.013171066530048847, -0.005456478334963322, -0.000800545618403703, 0.06329421699047089, -0.022385098040103912, -0.00322969863191247, 0.06706491112709045, -0.046550676226615906, 0.06126434728503227, -0.004244114272296429, -0.009011546149849892, 0.019765133038163185, -0.012262200936675072, -0.0037941380869597197, -0.012223133817315102, 0.06752893328666687, -0.0063244374468922615, -0.017897095531225204, -0.002689176704734564, 0.012348088435828686, -0.05291532725095749, -0.12472029775381088, -0.007683944422751665, -0.012630771845579147, -0.00032599171390756965, -0.06619144231081009, -0.07313303649425507, -0.031441666185855865, -0.039843324571847916, -0.04073844477534294, 0.001387257594615221, 0.031997695565223694, -0.004375803750008345, 0.007275755517184734, 0.03285742551088333, -0.000852733151987195, 0.011037217453122139, -0.06448384374380112, 0.009066944010555744, -0.020171228796243668, 0.013074535876512527, 0.049121610820293427, 0.00025295294472016394, 0.048182226717472076, 0.04152413085103035, 0.0917634442448616, 0.10010915994644165, 0.03548859804868698, 0.013197450898587704, 0.0029082533437758684, -0.0416613332927227, 0.0013416280271485448, 0.04073866456747055, 0.014824043959379196, -0.01225171610713005, -0.005874080117791891, 0.0010183242848142982, 0.006961850915104151, 0.06789807975292206, 0.016388900578022003, 0.0008932080818340182, 0.001778951147571206, 0.009391780942678452, 0.009567136876285076, -0.04602200165390968, -0.004594798199832439, 0.020674576982855797, -0.008121702820062637, 0.007937245070934296, 0.016899460926651955, 0.0839364230632782, 0.0054790545254945755, -0.0022348181810230017, 0.007506352383643389, -0.043506693094968796, 0.006418893579393625, 0.018397171050310135, -0.024819346144795418, -0.022947493940591812, -0.0018083791946992278, 0.0001558273215778172, -0.007878939621150494, 0.08476664125919342, 0.031023358926177025, -0.01093913521617651, -0.0045547871850430965, -0.0016931622521951795, -0.14531159400939941, 0.04652543365955353, 0.0013476567110046744, -0.07101965695619583, 0.02698277123272419, -0.00036421528784558177, 0.03589615970849991, -0.032342519611120224, -0.001371484831906855, -0.04250318929553032, -0.0034781794529408216, -0.0016090084100142121, 0.09996823221445084, -0.021492403000593185, -0.02414746955037117, 0.0005764175439253449, -0.010111531242728233, 0.00018419545085635036, -0.0033961532171815634, -0.0096285967156291, 0.03285318613052368, 0.04248035326600075, 0.02316875383257866, 0.06026811525225639, -0.011858263052999973, 0.024930613115429878, 0.008534815162420273, 0.05194813013076782, -0.007335222791880369, 0.030473696067929268, 0.0030430445913225412, -0.000516896543558687, -0.001482536201365292, 0.02014976181089878, -0.0035140570253133774, 0.038657695055007935, 0.019324172288179398, -0.009567213244736195, -0.01088794693350792, -0.024709399789571762, 0.001182150561362505, 0.0006072044488973916, -0.006950869224965572, 0.0024681929498910904, 0.004040537867695093, 0.0078986631706357, 0.019794698804616928, 0.01209986675530672, -0.14338906109333038, 0.007138155400753021, -0.01753271371126175, 0.00018508867651689798, 0.008288388140499592, 2.0435858459677547e-05, 0.021747536957263947, 0.023133769631385803, 0.020779021084308624, -0.03066561557352543, -0.014385402202606201, 0.007712927181273699, 0.0005569797358475626, 0.004057337064296007, 0.06414590775966644, 0.010707263834774494, -0.040578898042440414, 0.005791878327727318, 0.010393747128546238, -0.017176495864987373, -0.07318537682294846, -0.046375297009944916, -0.01415962539613247, 0.015462575480341911, -0.0917077362537384, 0.06678039580583572, -0.010957380756735802, 0.11456289887428284, -0.024097206071019173, -0.0020795210730284452, 0.04045391455292702, -0.0021864690352231264, 0.0018841504352167249, -0.007357314694672823, -0.12205787003040314, 0.0575392059981823, -0.017108742147684097, 0.0024076367262750864, -0.010788003914058208, -0.00359379593282938, 0.009140919893980026, 0.0020253241527825594, 0.014078868553042412, -0.007003842853009701, -0.02846800535917282, 0.009964349679648876, -0.10145753622055054, 0.016052715480327606, -0.013611027039587498, -0.01293004211038351, 0.03423381969332695, -0.0051689716055989265, -0.007278838194906712, -0.0018980416934937239, 0.007836001925170422, 0.16790567338466644, 0.010792295448482037, -0.02094963937997818, -0.08075296878814697, -0.037993185222148895, -0.052480995655059814, -0.026204898953437805, 0.007362688891589642, 0.014586571604013443, 0.009127425029873848, -0.010813000611960888, -0.15669366717338562, 0.03690202161669731, 0.019739583134651184, -0.012087946757674217, -0.02770879864692688, 0.03752385079860687, 0.07122442871332169, -0.007426009979099035, -0.03601256012916565, 0.03952217474579811, -0.015012236312031746, 0.03155921399593353, -0.13032664358615875, 0.1600874662399292, -0.02415359392762184, -0.017633071169257164, 0.012627018615603447, 0.021662026643753052, 0.003541249083355069, -0.051020245999097824, 0.01148495264351368, -0.041564058512449265, -0.04291853308677673, 0.00215087435208261, -0.11456901580095291, -0.0425846166908741, 0.008650599978864193, -0.02949083410203457, 0.006798249669373035, -0.03024865686893463, -0.015093792229890823, -0.05639670044183731, -0.021620862185955048, -0.059841495007276535, -0.022244444116950035, 0.003444820875301957, -0.0429564043879509, 0.018054254353046417, -0.004891116637736559, -0.025260990485548973, 0.06252143532037735, 0.00034027438960038126, 0.005672262515872717, -0.017360083758831024, 0.001533552655018866, 0.1153225302696228, 0.023573020473122597, -0.026702439412474632, 0.038318417966365814, 0.024855278432369232, -0.03045649640262127, 0.015382267534732819, 0.017001500353217125, -0.0015683294041082263, 0.017785828560590744, -0.011520267464220524, 0.003009646898135543, -0.010995952412486076, 0.03945397585630417, 0.00024065087200142443, 0.012523528188467026, 0.00028468898381106555, -0.025185709819197655, 0.07770320028066635, -0.004897051490843296, -0.09140690416097641, 0.05707649141550064, 0.006738176103681326, 0.026999041438102722, -0.0016950499266386032, 0.06665065139532089, 0.004677888005971909, -0.002569757867604494, -0.004210263025015593, -0.03753741458058357, -0.013686380349099636, -0.13314558565616608, -0.0030083206947892904, 0.0101370420306921, 0.014752950519323349, 0.08384871482849121, 0.04555479809641838, -0.016703395172953606, -0.05903373286128044, -0.044861942529678345, 0.00037293494096957147, -0.04358583316206932, 0.007703780196607113, 0.009087832644581795, -0.00676462659612298, -0.016775934025645256, 0.023479638621211052, 0.029306860640645027, -0.0012774438364431262, -0.028553910553455353, -0.0352836437523365, 0.12815330922603607, 0.04856380447745323, 0.010750837624073029, 0.08394094556570053, 0.011273620650172234, 0.13138508796691895, 0.003135478589683771, -0.002937913406640291, -0.010556046850979328, -0.1461666077375412, -0.019503219053149223, -0.003587687388062477, 0.009599425829946995, -0.03483222797513008, 0.03860071301460266, 0.02013031207025051, 0.028760837391018867, 0.010715088807046413, 0.028836917132139206, 0.004093140363693237, 0.02005310356616974, 0.027271628379821777, -0.000658693490549922, -0.033699557185173035, -0.059954240918159485, -0.007230325601994991, -0.043142110109329224, -0.00962083600461483, -0.012090849690139294, 0.03835412859916687, -0.0003924237680621445, 0.00045790488366037607, 0.0050728763453662395, -0.05206892266869545, 0.0021625699009746313, 0.002682153368368745, -0.03188180923461914, -0.0011614650720730424, 0.0006023325840942562, -0.0019967618864029646, -0.09218623489141464, 0.09806189686059952, -0.017020326107740402, 0.0013698937837034464, 0.013267485424876213, -0.01236817892640829, -0.14612793922424316, 0.03436213359236717, -0.009095831774175167, -0.039626460522413254, -0.013333517126739025, 0.0018668662523850799, -0.016819003969430923, -0.04454794153571129, 0.00033706362592056394, -0.01909065805375576, -0.04305367171764374, 0.004686505068093538, -0.03824974596500397, -0.043898969888687134, -0.06665215641260147, -0.01687815599143505, 0.015911469236016273, 5.351216532289982e-05, 0.11244027316570282, -0.00644442206248641, -0.04646662622690201, -0.002619558246806264, 0.04444903880357742, 0.12201514840126038, 0.010375989601016045, 0.04351847991347313, 0.018507113680243492, 0.04293844476342201, -0.08495970815420151, 0.036498747766017914, 0.042269352823495865, -0.0024100018199533224, -0.04165258631110191, -0.04806559532880783, 0.008730735629796982, 0.03645243123173714, -0.00039751167059876025, 0.010563720017671585, -0.01425118651241064, -0.006394566968083382, -0.0003018772986251861, 0.00018762811669148505, -0.003631687955930829, 0.0035644243471324444, 0.1092342808842659, 0.014157520607113838, 0.028052063658833504, -0.05222403630614281, -0.10089143365621567, -0.026341207325458527, -0.02406957745552063, -0.0019520062487572432, -0.033260613679885864, -0.0002866688882932067, 0.03476439043879509, 0.06708276271820068, -0.0022107111290097237, -0.007868589833378792, -0.06606870889663696, -0.0011869743466377258, 0.0007026843377389014, -0.07788731157779694, 0.07015715539455414, 0.01325170136988163, 0.016182996332645416, -0.015387781895697117, -0.08609835803508759, 0.03670111671090126, -0.07281075417995453, 0.0159313902258873, -0.018510866910219193, -0.015674293041229248, -0.006566074211150408, 0.009352841414511204, -0.04953208938241005, 0.04818602278828621, 0.027591153979301453, -0.0004822415066882968, -0.023111989721655846, -0.0010218912502750754, 0.0350237600505352, 0.008220249786973, -0.017356211319565773, -0.04290126636624336, 0.00875760242342949, 0.002275831298902631, 0.026735272258520126, -0.005537169519811869, 0.033954765647649765, -0.045698750764131546, 0.05743629112839699, 0.03380494937300682, -0.12042410671710968, -0.0018299074145033956, -0.04643753170967102, 0.043463774025440216, 0.011805749498307705, 0.0932508185505867, -0.0023011555895209312, -0.01835666596889496, 0.0005184095352888107, 0.0074603985995054245, -0.002863718196749687, 0.0004903157241642475, 0.01399854477494955, -0.060539279133081436, -0.02038407325744629, -0.052296627312898636, -0.15499357879161835, -0.06458506733179092, 0.015548567287623882, 0.01999633014202118, -0.036342915147542953, 0.006704783067107201, -0.13323979079723358, 0.027560096234083176, 0.008850261569023132, -0.025227805599570274, -0.011286405846476555, 0.042342159897089005, -0.013491973280906677, 0.06154234707355499, -0.022803297266364098, 0.027578257024288177, -0.06836282461881638, 0.017900127917528152, -0.06587554514408112, 0.1640283614397049, 0.0013537168269976974, -0.03628595545887947, 0.048769865185022354, -0.02432009018957615, -0.0382644459605217, -0.02519717812538147, 0.011026156134903431, 0.005201833322644234, -0.0027653705328702927, 0.001110657467506826, 0.03058467246592045, -0.02502460777759552, -0.004959498532116413, 0.002582837361842394, -0.0035857013426721096, -0.04483937472105026, -0.009197406470775604, -0.025922702625393867, 0.0027069777715951204, 0.07694163918495178, 0.09675545990467072, -0.04319481924176216, -0.026645665988326073, -0.05702845752239227, 0.016045253723859787, -0.014182484708726406, 0.13632208108901978, -0.021834667772054672, -0.0014771041460335255, 0.010295309126377106, 0.03701918572187424, 0.1381395310163498, 0.008825189433991909, 0.0699775218963623, -0.01047273725271225, -0.05426793545484543, -0.06220127269625664, 0.03348333016037941, -0.004885565955191851, 0.08865807205438614, -0.01583140157163143, -0.005070621147751808, 0.011818687431514263, -0.028625518083572388, -0.005750976502895355, 0.0075773755088448524, 0.0919732004404068, -0.002198245143517852, -0.006051817908883095, 0.004567941650748253, -0.00603511044755578, -0.042486075311899185, 0.19634626805782318, 0.008842428214848042, -0.0011013469193130732, -0.005244515836238861, -0.004170076455920935, -0.04403817653656006, 0.00396912032738328, 0.008348891511559486, 0.007532151881605387, 0.014424215070903301, -0.03898874670267105, 0.007781099062412977, 0.07245801389217377, 0.002165556186810136, 0.2104017436504364, 0.1604994386434555, -0.0026304544880986214, 0.002100599929690361, 0.011162101291120052, -0.012936157174408436, 0.007843182422220707, 0.0025272779166698456, 0.02598448470234871, -0.019763197749853134, -0.0916697308421135, 0.0122487498447299, 0.014393799006938934, -0.02461099810898304, 0.03156360983848572, -0.04822372645139694, 0.07638843357563019, 0.020253075286746025, 0.0682009682059288, 0.03439193591475487, -0.009599242359399796, 0.020171789452433586, 0.012183310464024544, 0.002142240758985281, -0.002687918022274971, -0.15638549625873566, -0.024169746786355972, -0.016400117427110672, 0.01866055093705654, -0.09142417460680008, 0.03010244108736515, -0.05858704075217247, 0.018729984760284424, 0.00183092278894037, 0.05499187856912613, -0.008466362953186035, -0.061766985803842545, -0.002406308427453041, -0.004169910680502653, 0.002218149369582534, -0.11806195229291916, -0.002480853581801057, 0.016424158588051796, -0.012639087624847889, -0.01408260129392147, -0.01099417544901371, 0.009266424924135208, 0.0007466915994882584, 0.00030874990625306964, -0.07379043102264404, -0.0009689657017588615, -0.008631587959825993, -0.0035136679653078318, -0.011549071408808231, -0.0059485118836164474, 0.00012821693962905556, 0.013498893938958645, 0.03588203713297844, -0.02644588239490986, 0.00114786671474576, -0.01830594800412655, -0.007731582969427109, -0.03497711941599846, 0.016389455646276474, 0.01277845073491335, 0.08886346220970154, -0.08164075016975403, 0.001447360380552709, 0.03781285509467125, -0.0847131535410881, -0.00011096548405475914, -0.0554974190890789, 0.014491304755210876, -0.03939799219369888, -0.005500894505530596, -0.032322704792022705, -0.02514953911304474, 0.03362162783741951, 0.017421802505850792, -0.0006581777706742287, 0.016424404457211494, 0.0006117492448538542, -0.06378907710313797, -0.032554734498262405, 0.07654237002134323, 0.028388241305947304, 0.0032927975989878178, 0.028301838785409927, 0.026385655626654625, -0.02965058945119381, -0.04706696793437004, 0.03750242292881012, 0.1403590738773346, -0.0006024062167853117, -0.030677972361445427, 0.010169905610382557, 0.012507571838796139, -0.02346656285226345, 0.008528390899300575, -0.01340678334236145, 0.07331323623657227, -0.025709880515933037, 0.00042923001456074417, -0.0002779827336780727, 0.009679703041911125, -0.03437109664082527, 0.19045910239219666, 0.011064992286264896, 0.03562427684664726, -0.16365410387516022, -0.0037593713495880365, -0.021922890096902847, -0.026025395840406418, -0.0007703155279159546, -0.04968823865056038, -9.290442540077493e-05, -0.08224872499704361, 0.04084751754999161, 0.06148111820220947, -0.018648160621523857, -0.0034983805380761623, 0.015826405957341194, -6.828972982475534e-05, -0.024117734283208847, 0.03847747668623924, -0.001414632541127503, 0.01581263728439808, -0.037034131586551666, 0.02216995693743229, -0.016589131206274033, -0.014687970280647278, 0.012795036658644676, -0.009204964153468609, 0.00022539040946867317, 0.11568952351808548, -0.021332599222660065, -0.03940153867006302, 0.010599364526569843, -0.017116719856858253, 0.0459880456328392, -0.0006968237576074898, 0.007346934173256159, 0.03431433439254761, 0.0024865202140063047, -0.0030169233214110136, 0.009233947843313217, 0.010923032648861408, -0.01614929921925068, 0.023156680166721344, 0.0020812484435737133, -0.0036086374893784523, -0.004606260452419519, 0.012043634429574013, 0.035258907824754715, -0.024310944601893425, -0.009570814669132233, -0.0034347993787378073, -0.0029580374248325825, 0.026989834383130074, 0.04781259223818779, 0.008218767121434212, -0.0073919352144002914, 0.006989404559135437, 0.014063606970012188, 0.001326013240031898, -0.013160341419279575, -0.005182589869946241, -0.050782788544893265, 0.023173991590738297, -0.05885937064886093, -0.13905739784240723, -0.029490742832422256, -0.0566718690097332, 0.006910337600857019, 0.004866437986493111, -0.013212038204073906, -0.002625213237479329, -0.021700065582990646, 0.07320429384708405, -0.057375237345695496, 0.02422933280467987, -0.026632260531187057, -0.03260452672839165, -0.042108818888664246, 0.054962120950222015, 0.04837492108345032, -0.010678108781576157, 0.024917636066675186, 0.01812555268406868, 0.040120698511600494, 0.016940170899033546, -0.092657171189785, -0.041443921625614166, -0.03260247781872749, -0.01880168728530407, 0.02640770748257637, -0.004420516546815634, -0.002616685815155506, -0.08032811433076859, 0.0003877861308865249, 0.018679281696677208, 0.10533161461353302, -0.020713668316602707, 0.07856928557157516, 0.0009786596056073904, -0.07827212661504745, -0.008771066553890705, 0.019792957231402397, 0.014649315737187862, -0.019508009776473045, 0.0008910470642149448, -0.08770991116762161, -0.006854305975139141, -0.021972376853227615, -0.012170573696494102, -0.025222525000572205, 0.16217219829559326, -0.11968441307544708, 0.0008029031450860202, 0.03137275576591492, 0.0844917893409729, 0.026465216651558876, 0.03415984287858009, 0.0029331098776310682, -0.03960972651839256, -0.028244422748684883, -0.006499679759144783, -0.024392547085881233, -0.03556405007839203, -0.007026265375316143, 0.010377933271229267, -0.014629540033638477, -0.00497090071439743, -0.03044537641108036, -0.046050652861595154, 0.00885068066418171, 0.013133389875292778, -0.005604836158454418, -0.08861537277698517, 0.002616657642647624, -0.003449344076216221, -0.027150306850671768, 0.061589524149894714, 0.007544361520558596, 0.025470785796642303, -0.02688051573932171, 0.003550560912117362, -0.030393363907933235, 0.014289687387645245, -0.010548201389610767, 0.05238758772611618, -0.058506835252046585, -0.01175701804459095, -0.013399052433669567, 0.020017525181174278, -0.0006394547526724637, 0.16248221695423126, 0.09039713442325592, -0.028295408934354782, 0.026810593903064728, -0.04864612594246864, 0.008907977491617203, 0.05591561645269394, -0.027466431260108948, -0.01602000743150711, 0.00823682826012373, -0.00563445221632719, 0.010356663726270199, 0.008600970730185509, 0.0054582254961133, -0.020696984604001045, 0.014944794587790966, 0.07397497445344925, 0.03933217003941536, 0.06157953292131424, 0.01990300789475441, 0.09925730526447296, 0.055965594947338104, 0.030658988282084465, 0.09280996024608612, 0.038419194519519806, -0.15234731137752533, -0.11178568005561829, -0.018837550655007362, 0.06939166784286499, -0.20737995207309723, 0.05328720808029175, 0.010149690322577953, 0.11311937123537064, 0.04745527356863022, 0.03563402220606804, 0.0030249750707298517, -0.018739929422736168, -0.015197979286313057, 0.03242141008377075, -0.08584625273942947, -0.12107311934232712, 0.006828294601291418, -0.013080720789730549, -0.009778580628335476, -0.002514589112251997, -0.00956083182245493, -0.007347956299781799, -0.0008249290403909981, -0.0010134816402569413, 0.025873471051454544, 0.00850385706871748, -0.006874109152704477, 0.006248290650546551, -0.045207537710666656, -0.0009259129292331636, -0.00016716390382498503, 0.04792173206806183, 0.020013989880681038, 0.004846288822591305, -0.003029576735571027, 0.012722627259790897, -0.011843890883028507, -0.04011852666735649, -0.01781332865357399, 0.021907292306423187, 0.01731489971280098, -0.056357938796281815, -0.001755732810124755, 0.040328092873096466, -0.17720471322536469, -0.0009857388213276863, -0.03289851173758507, 0.0029644002206623554, 0.023852480575442314, 0.016814153641462326, -0.024451836943626404, 0.005351416766643524, 0.001302481396123767, 0.020102212205529213, -0.000692866975441575, 0.03904159739613533, -0.002157309791073203, -0.014382758177816868, 0.006196900736540556, -0.00415873434394598, 0.036376841366291046, 0.009662032127380371, 0.14167320728302002, 0.0091716218739748, -0.02505551651120186, -0.11870361119508743, 0.05134432017803192, 0.031753022223711014, -0.001061795512214303, -0.018286628648638725, 0.0028119892813265324, -0.012998745776712894, 0.015374449081718922, 0.010473604314029217, 0.03043915331363678, -0.015407891012728214, -0.03400706127285957, 0.00012773630442097783, 0.00012636036262847483, -0.03538209944963455, 0.012818771414458752, 0.12452613562345505, -0.00808822363615036, 0.016384907066822052, -0.026793375611305237, -0.06947942078113556, -0.05352642014622688, -3.065679993596859e-05, -0.0007501197978854179, -0.015947796404361725, 3.1603682145942e-05, 0.023562662303447723, 0.02542494423687458, -0.02687826380133629, -0.03661687299609184, -0.00019525706011336297, 0.012135511264204979, 0.0017545747105032206, -0.012982703745365143, 0.00018897585687227547, 0.038890961557626724, 0.027356313541531563, -0.025968672707676888, 0.024486921727657318, -0.02826101705431938, -0.0650864914059639, -0.018695125356316566, -0.13179899752140045, 0.000297106453217566, -0.0010824267519637942, 0.002270005876198411, -0.024916520342230797, 0.007615437265485525, -0.04759124666452408, -0.019669193774461746, 0.09889223426580429, 0.004189854487776756, -0.013632542453706264, -0.0007760284934192896, -0.026463912799954414, -0.06296684592962265, 0.006688953842967749, 0.022693784907460213, 0.020438408479094505, -0.00036426979932002723, 0.005440190434455872, -0.0036309086717665195, 0.14438441395759583, 0.015814529731869698, 0.006122914142906666, -0.011456206440925598, -0.02929205261170864, 0.02705290913581848, 0.007067963946610689, 0.09383674710988998, 0.05438917130231857, -0.011879057623445988, -0.01650993339717388, -0.008036809042096138, 0.008603110909461975, 0.07711467891931534, 0.013789454475045204, 0.0005302408244460821, 0.029451267793774605, -0.06644263118505478, -0.15567553043365479, -0.03157198429107666, 0.028594505041837692, 0.01988273486495018, -0.0011728337267413735, 0.005527454428374767, -0.023016734048724174, 0.009298892691731453, 0.082683265209198, -0.09216828644275665, -0.04381515458226204, -0.00906041357666254, -0.015252255834639072, -0.011102445423603058, 0.02209978923201561, 0.025475241243839264, -0.015211696736514568, -0.003522792598232627, -0.009384051896631718, 0.04605507850646973, 0.0029357573948800564, -0.17484915256500244, 0.018092786893248558, 0.028652532026171684, 0.009761913679540157, 0.0058389585465192795, 0.005338709335774183, -0.010197864845395088, 0.022712938487529755, -0.0004904670058749616, -0.07003334909677505, 0.0587441585958004, -0.02991221658885479, -0.02917490154504776, 0.003892601700499654, 0.012310100719332695, -0.010113297030329704, -0.01359877921640873, -0.0020756497979164124, -0.06279479712247849, 0.005332238972187042, -0.014828491024672985, -0.005506914108991623, -0.012180262245237827, 0.07601779699325562, -0.11338220536708832, 0.1250581443309784, -0.023224635049700737, 0.005439223255962133, 0.023659830912947655, 0.026951812207698822, -0.022251388058066368, 0.022355657070875168, -0.08142324537038803, 0.020685676485300064, -0.01664697751402855, -0.03251052647829056, -0.0001891615684144199, -0.015143651515245438, 0.00790463201701641, 0.02062075585126877, -0.011360338889062405, -0.0027529760263860226, -0.0456276535987854, -0.09111593663692474, 0.010645899921655655, -0.0014054140774533153, 0.02130107395350933, 0.030170544981956482, -0.02775203064084053, -0.0029969278257340193, -0.013394390232861042, 0.020177999511361122, 0.05792831629514694, -0.0034507669042795897, -0.02893560379743576, -0.15740865468978882, -0.01106127630919218, -0.03206726536154747, -0.0018558462616056204, 0.044013336300849915, -0.020778993144631386, 0.03495636209845543, 0.03026718460023403, 0.03307345509529114, -0.003064294345676899, 0.1039147824048996, 0.16674470901489258, 0.006846808362752199, -0.07768817245960236, -0.050250906497240067, -0.0015413500368595123, -0.0046066781505942345, -0.06419459730386734, 0.0014963203575462103, -0.009212399832904339, -0.018176263198256493, 0.001977125881239772, 0.00409303605556488, -0.018748093396425247, 0.019510187208652496, 0.00037631893064826727, -0.030825335532426834, -0.02345295622944832, 0.02029508352279663, -0.017377221956849098, 0.002726971870288253, -0.03636317700147629, 0.0042581758461892605, 0.003946540877223015, 0.006765842903405428, -0.03677380084991455, -0.008314554579555988, -0.10901079326868057, 0.03486720472574234, -0.002761801006272435, -0.039398305118083954, -0.016600128263235092, 0.012879611924290657, -0.005384760443121195, -0.010831385850906372, -0.014172263443470001, -0.013201805762946606, -0.011680047027766705, -0.01163654774427414, -0.0597686693072319, -0.0337698757648468, 0.09315718710422516, -0.03152254968881607, 0.03310737758874893, -0.026724092662334442, -0.08826553821563721, 0.0017398573691025376, -0.00029864799580536783, -0.011559846810996532, -0.03546635061502457, 0.004836325068026781, -0.08660940825939178, 0.038659196346998215, 0.005613436922430992, -0.0025437918957322836, -0.0009489618241786957, 0.004921965766698122, 0.044547583907842636, -0.06839244067668915, 0.01181704830378294, -0.006693336647003889, 0.0018873484805226326, 0.03585050627589226, 0.002352470764890313, 0.002517144428566098, -0.0022189635783433914, -0.042079534381628036, 0.0027832635678350925, -0.005388887599110603, -0.030686115846037865, -0.0006187162944115698, 0.059906136244535446, 0.006439435761421919, 0.001484862994402647, -0.015850532799959183, 0.02504279650747776, 0.010861244983971119, 0.0014431189047172666, 0.11675997078418732, 0.001679274719208479, 0.012623296119272709, -0.004369651433080435, -0.08023107796907425, -0.03505018353462219, 0.013182266615331173, -0.0005644014454446733, 0.07832325249910355, 0.01210481021553278, 0.022810203954577446, -0.01651609316468239, 0.01992160454392433, 0.062073081731796265, 0.025856543332338333, -0.00023292102559935302, -0.05496794357895851, -0.0011004642583429813, 0.07999202609062195, 0.04011670872569084, 0.014007330872118473, 0.009361648932099342, 0.015067423693835735, 0.008822271600365639, -1.6079822671599686e-05, -0.0003294895577710122, -0.005308948922902346, 0.003493283176794648, 0.032729506492614746, -0.015975862741470337, 0.010250259190797806, -0.07238570600748062, 0.033242832869291306, -0.009067164734005928, 0.0006137430900707841, -0.0007913547451607883, -0.02830120176076889, -0.0001775525597622618, -0.0033455672673881054, 0.01674419455230236, -0.0705835372209549, -0.005575842224061489, -0.034060895442962646, -0.019331062212586403, 0.0005751143908128142, -0.07981719076633453, -0.030152561143040657, 0.01641746796667576, -0.008596662431955338, 0.006664434913545847, -0.014828831888735294, 0.0032087932340800762, 0.018860913813114166, 0.023690706118941307, -0.005423409398645163, -0.08079713582992554, 0.09203816950321198, -0.15080192685127258, -0.0693126693367958, 0.015564482659101486, -0.0003977054147981107, 0.0026519440580159426, 0.017434613779187202, 0.014321576803922653, 0.064304880797863, 0.019748495891690254, 0.0011474993079900742, -0.011945188976824284, 0.035579580813646317, 0.052748288959264755, 0.00035930488957092166, 0.006398916710168123, 0.011714749969542027, 0.0040855673141777515, 0.013927854597568512, -0.034392405301332474, -0.0006596018210984766, -0.078335240483284, 0.01638125069439411, 0.034343522042036057, 0.03683362156152725, -0.009871043264865875, -0.053423747420310974, 0.00915467832237482, 0.0018532377434894443, -0.07978837192058563, -0.0015575849683955312, -0.02570316381752491, 0.013453792780637741, 0.007390417158603668, -0.02003549598157406, -0.014253836125135422, -0.059821490198373795, 0.01296711154282093, -0.005614461377263069, 0.006564262323081493, -0.019413502886891365, -0.009298557415604591, 0.11380168050527573, -0.0034652063623070717, 0.024138376116752625, 0.016127876937389374, -0.022896388545632362, -0.12581394612789154, 0.01030116155743599, -0.005465333815664053, 0.17241591215133667, 0.0176678616553545, -0.016305601224303246, -0.01721174269914627, 0.006095873657613993, -0.029051022604107857, -0.03743389621376991, 0.021959472447633743, 0.021573903039097786, 0.053302403539419174, -0.027171874418854713, 0.03634878620505333, 0.016032181680202484, 0.012877631932497025, 0.03413960710167885, 0.005625386256724596, -0.00893899705260992, -0.04919414222240448, -0.07964105904102325, -0.0022713125217705965, 0.001040024682879448, 0.0697135403752327, -0.010696714743971825, -0.08518757671117783, -0.028328808024525642, 0.013750612735748291, 0.0201739314943552, -0.018611369654536247, -0.07456739991903305, 0.02420586161315441, 0.007642708718776703, -0.004908427130430937, -0.010427567176520824, -0.009094804525375366, 0.0006645643734373152, -0.036170843988657, 0.012710627168416977, 0.025574347004294395, 0.0244744960218668, 0.007712449878454208, -0.013142053037881851, 0.022918475791811943, -0.02037947252392769, 0.00014044265844859183, -0.0012279058573767543, -0.012383310124278069, -0.009287964552640915, 0.03117172047495842, 0.0003813690273091197, 0.07886265963315964, -0.05565235763788223, -0.021799985319375992, -0.0015253847232088447, 0.020564252510666847, 0.03240026533603668, -0.002958927769213915, -8.039933163672686e-05, -0.03041953034698963, -0.01725318841636181, 0.04915697127580643, 0.01075046043843031, -0.02340730093419552, -0.07021209597587585, 0.004830440040677786, -0.05471364036202431, 0.001493365503847599, -0.007103704381734133, -0.054170817136764526, -0.019121326506137848, 0.021410705521702766, 0.00034613272873684764, -0.0014388429699465632, -0.06968258321285248, 0.006566434632986784, 0.016225121915340424, -0.146831214427948, -0.02603265270590782, -0.003129365621134639, -0.0030772851314395666, -0.01910204067826271, 0.0009230343857780099, -0.0229498203843832, 0.011486888863146305, 0.006551247090101242, 0.0070651923306286335, -0.0015768160810694098, 0.0024922138545662165, 0.009272565133869648, -0.012006998062133789, 0.014306632801890373, 0.051140476018190384, -0.0016803048783913255, 0.007955921813845634, -0.0602213516831398, -0.013751198537647724, 0.008151314221322536, -0.004000811371952295, 0.019324202090501785, 0.0282877329736948, -0.03051375411450863, -0.005738610401749611, 0.20339886844158173, -0.10123791545629501, -0.003864624071866274, -0.008548520505428314, -0.03446154668927193, -0.01952134259045124, -0.009370170533657074, -0.002910333452746272, 0.035997770726680756, 0.0008862767135724425, 0.013431605882942677, 0.0029985010623931885, 0.05384331941604614, 0.032945528626441956, 0.09609243273735046, -0.010296734981238842, 0.006425104569643736, -0.004620505962520838, 0.0002367031411267817, -0.016803201287984848, 0.06506005674600601, 0.012500331737101078, -0.05652608722448349, 0.0024125631898641586, -0.022056622430682182, -0.0016707435715943575, -0.0005337988841347396, 0.014763188548386097, 0.0523097924888134, -0.062493033707141876, 0.011130684055387974, -0.010543488897383213, 0.009874232113361359, -0.021756060421466827, -0.008400778286159039, -0.013234497047960758, -0.010603128001093864, 0.025122178718447685, 0.0021009778138250113, -0.014529109001159668, 0.10283071547746658, -0.0004497528134379536, 0.029976937919855118, -0.004918878898024559, 0.003458749270066619, 0.006065032910555601, 0.10786692053079605, 0.013743314892053604, 0.0003143796056974679, 0.024748245254158974, -0.0004623853601515293, 0.024294843897223473, 0.0007116901688277721, 0.0039998264983296394, -0.016878291964530945, -0.031119845807552338, -0.024597782641649246, 0.02865719608962536, -0.07361308485269547, -0.008429918438196182, 0.04647757112979889, 0.000965698214713484, 0.026107830926775932, -0.008122460916638374, 0.00041691618389450014, -0.0396789088845253, -0.009062718600034714, 0.08759903907775879, -0.038327060639858246, 0.0002609241346362978, -0.01554007176309824, 0.0002571023942437023, -0.04039423540234566, -0.00026168557815253735, 0.0003378168330527842, 0.03335303068161011, 0.001868303632363677, -0.06445035338401794, -0.004470753483474255, 0.014578805305063725, 0.01569395698606968, -0.0186990424990654, 0.012666860595345497, -0.007149956189095974, 0.0016840290045365691, -0.013228890486061573, 0.0003003160236403346, 0.0018320676172152162, 0.010501645505428314, -0.049516238272190094, -0.014119276776909828, 0.003713719779625535, -0.0179043710231781, 0.0016635865904390812, -0.014807488769292831, 0.0183454230427742, 0.0023395547177642584, -0.06012922525405884, -0.012654128484427929, -0.002828943310305476, -0.003470631083473563, 0.04939069226384163, -0.02071031741797924, 0.008354255929589272, -0.09460769593715668, 0.05271727219223976, -0.11150842905044556, -0.03104972653090954, 0.07251188904047012, 0.0169198177754879, 0.008719620294868946, 0.007091861218214035, 0.004591358359903097, 0.025743963196873665, 0.00499984435737133, -0.007426420226693153, -0.09629090875387192, 0.012453749775886536, 0.039915964007377625, -0.06323949247598648, 0.0034279287792742252, 0.0017602829029783607, 0.0016919808695092797, -0.01725454069674015, 0.007394207641482353, -0.005905145313590765, -0.009964979253709316, 0.02286812849342823, 0.039566610008478165, 0.02875521034002304, 0.03707230091094971, -0.17257621884346008, 0.010965202003717422, 0.0036305508110672235, -0.11187548190355301, 0.004889850504696369, 0.012706831097602844, 0.007746576797217131, -0.019701946526765823, -0.06150510162115097, 0.023318355903029442, -0.037529751658439636, 0.01604914478957653, -0.005388122983276844, -0.0009763388079591095, -0.029279399663209915, -0.008018722757697105, 0.0194074884057045, 0.1648166924715042, -0.004209495615214109, 0.032636575400829315, -0.00476395059376955, -0.03248576074838638, 0.05098073184490204, -0.005298042204231024, 0.034726645797491074, 0.019390927627682686, -0.008087791502475739, 0.008110187016427517, -0.10769715905189514, 0.034231968224048615, -0.020149387419223785, 0.08678486198186874, 0.01603296399116516, -0.020126529037952423, -0.04226941242814064, -0.04135863110423088, 0.007408281788229942, 0.016010340303182602, -0.06850922852754593, 0.0007404398638755083, -0.0015433408552780747, -0.011133833788335323, 0.005383537150919437, -0.026493864133954048, 0.0016573938773944974, -0.011163447983562946, -0.004868520889431238, 0.004325822461396456, -0.04034543037414551, -0.02698778547346592, 2.0300037704146234e-06, 0.00427585281431675, -0.013907818123698235, -0.024747209623456, 0.0025977406185120344, 0.00817149132490158, -0.017636366188526154, 0.018426261842250824, -0.001988147385418415, -0.06589139252901077, -0.005319532006978989, 0.09481138736009598, 0.006960521452128887, 0.03383663296699524, 0.0035480449441820383, -0.012271066196262836, 0.004412398207932711, 0.010084097273647785, 0.01694643124938011, 0.024593476206064224, -0.07965598255395889, 0.11134815961122513, 0.0006173055153340101, -0.005122222937643528, 0.07004231959581375, -0.10110688954591751, -0.02456038072705269, 0.0028791320510208607, -0.06601011753082275, 0.011965924873948097, -0.00498202396556735, -0.13615617156028748, -0.016654523089528084, 0.019703492522239685, -0.01652809977531433, -0.033586081117391586, -0.11262655258178711, 0.047503259032964706, -0.012446560896933079, -0.015996769070625305, -0.02300059050321579, -0.0060512833297252655, 0.009559941478073597, 0.007166897412389517, 0.0037971017882227898, -0.003445475362241268, -0.06185959652066231, 0.07274496555328369, 0.042227718979120255, -0.041737087070941925, -0.023853600025177002, 0.00936417281627655, 0.005493757780641317, -0.11818009614944458, -0.04688957333564758, -0.00100129924248904, -0.01685638353228569, 0.008591103367507458, -0.010244808159768581, -0.02534020133316517, 0.008996988646686077, 0.05421094596385956, -0.031246298924088478, -0.030564123764634132, -0.04576660692691803, -0.01298273541033268, 0.0046663773246109486, -0.016763167455792427, 0.02803190052509308, 0.0050767408683896065, 0.0012302405666559935, -0.01201110240072012, 0.02479356713593006, -0.01373340468853712, 0.05248657613992691, 0.04099617153406143, 0.00335480528883636, -0.018130788579583168, 0.010515373200178146, -0.06796710938215256, -0.04328799247741699, -0.12534959614276886, 0.02222255989909172, 0.0006088320515118539, 0.00578210037201643, -0.009761895053088665, -0.026865050196647644, -0.008963788859546185, -0.006900481879711151, 0.037439994513988495, -0.01066798809915781, -0.011324778199195862, -0.020252101123332977, 0.00039674638537690043, 0.0038270477671176195, -0.06734401732683182, 0.005940386094152927, -0.020634498447179794, 0.026459259912371635, 0.0014427561545744538, -0.0037973641883581877, -0.0004404591745696962, -0.0033278975170105696, 0.05280708149075508, 0.002002759836614132, -0.001615874352864921, -0.002439669566228986, 8.085835725069046e-05, 0.0031094583682715893, 0.008939621038734913, -0.01743038184940815, -0.0005763257504440844, 0.051459137350320816, -0.003568155923858285, 0.0586077943444252, -0.057633038610219955, -0.0005782184889540076, -0.058289218693971634, 0.0069075641222298145, -0.003750280709937215, -0.001819917932152748, 0.017611507326364517, 0.13050082325935364, 0.004028978757560253, 0.005180531181395054, 0.0013401827309280634, 0.07042249292135239, -0.005191896576434374, -0.03009411320090294, 0.012732385657727718, -0.025382282212376595, 0.0074938880279660225, 0.01408534124493599, -0.0066286115907132626, -0.01310724951326847, 0.014633885584771633, -0.02871698886156082, 0.05192384496331215, 0.004024303052574396, 0.0013609308516606688, -0.012354045175015926, 0.00925121083855629, 0.0563383623957634, 0.023035231977701187, 0.004008455201983452, -0.01141047291457653, -0.0013942025834694505, -0.0028572306036949158, 0.0001759603328537196, 2.6396890461910516e-05, -0.03106403723359108, 0.041981443762779236, -0.05545017123222351, -0.03539301082491875, -0.006259921472519636, 0.010322486981749535, -0.055346500128507614, 0.008062103763222694, 0.011916388757526875, 0.0064952257089316845, -0.07200716435909271, -0.00027267212863080204, 0.0076463730074465275, 0.008992576971650124, 0.003405767260119319, 0.00010602809197735041, 0.12655533850193024, -0.16401684284210205, -0.0005767472321167588, 0.019070768728852272, -0.007774901110678911, 0.05605204030871391, -0.04432295635342598, 0.01636505126953125, 0.04796470329165459, 0.0075703212060034275, 0.06059299409389496, -0.037254296243190765, -0.004438546486198902, -0.0275095347315073, 0.05613377317786217, -0.05715184658765793, 0.05731796845793724, 0.016432102769613266, 0.008310615085065365, 0.019093500450253487, -0.03399408981204033, 0.00714672077447176, 0.038122013211250305, 0.008776978589594364, 0.005308559164404869, -0.021057162433862686, 0.006072306539863348, 0.01856313832104206, -0.016699889674782753, -0.0003823329461738467, 0.13580207526683807, -0.0989510715007782, 0.015004388056695461, 0.011701342649757862, -0.013919559307396412, -0.005121300462633371, 0.018520701676607132, 0.08659466356039047, 0.02070300094783306, -0.04782814905047417, -0.05645882710814476, 0.00022025563521310687, -0.007217521779239178, -0.016264749690890312, 0.0024627377279102802, 0.00834023579955101, 0.16066737473011017, 0.010916114784777164, -0.042947109788656235, -0.006316252984106541, -0.07755632698535919, -0.004162580706179142, -0.009265760891139507, 0.013649939559400082, -0.02319452166557312, -0.015762750059366226, 0.0889679491519928, -0.00896213948726654, 0.060599423944950104, -0.0030124541372060776, 0.0001286230981349945, -0.027523698285222054, -0.09639067202806473, 0.011713746003806591, 0.03853167966008186, 0.03278693929314613, -0.05021945387125015, 0.013272637501358986, -0.011539608240127563, 0.0023656326811760664, 0.0005934092332608998, 0.06347537040710449, -0.05707591027021408, -0.011075621470808983, -0.03091883286833763, -0.02200983464717865, 0.012572438456118107, -0.012497036717832088, -0.018256964161992073, -0.001850753091275692, 0.010986569337546825, 0.01370440237224102, -0.0076841446571052074, -0.00905068963766098, 0.000927372311707586, 0.016086485236883163, -0.007499200291931629, -0.02337247133255005, -0.03472108393907547, -0.02274332195520401, -0.015622017905116081, 0.021628964692354202, -0.02419518120586872, -0.01271151751279831, 0.005022866651415825, -0.009110844694077969, 0.008049297146499157, 0.00034860812593251467, 0.0026336712762713432, -0.011640406213700771, -0.001392235280945897, 0.06358795613050461, 0.011359945870935917, 0.010013438761234283, -0.0685192346572876, -0.007452836260199547, -0.009703285992145538, 0.02006351761519909, 0.03862348943948746, -0.055968429893255234, -0.001985142705962062, 0.004649130627512932, -0.0009463888127356768, 0.050722792744636536, -0.005209716968238354, 0.004460049793124199, -0.07111276686191559, 0.0013375667622312903, -0.08898577094078064, 0.01505003497004509, -0.005115795880556107, -0.017613140866160393, 0.0409676693379879, 0.008077344857156277, -0.00019762333249673247, -0.07457701116800308, -0.11099940538406372, 0.06383875757455826, -0.03017408959567547, -0.04979066923260689, 0.05126526206731796, 0.038336776196956635, 0.025218185037374496, -0.020836934447288513, 0.07943391054868698, 0.002711153356358409, -0.031401291489601135, 0.06561848521232605, 0.00799696147441864, -0.006240633316338062, 0.04487232118844986, 0.04770435765385628, 0.006356407422572374, -0.11649375408887863, 0.04563448578119278, 0.03854239359498024, -0.09679421037435532, 0.004310215823352337, -0.0027548051439225674, 0.1020907312631607, 0.025297673419117928, 0.037987120449543, 0.03497590497136116, 0.025266803801059723, 0.015740158036351204, -0.013533506542444229, 0.018755819648504257, -0.005621278192847967, 0.011282730847597122, 0.002708816435188055, 0.04274223744869232, 0.01931801438331604, 0.06309100240468979, 0.00945284403860569, 0.005107643082737923, -0.13612034916877747, 0.024314550682902336, 0.024846112355589867, -0.0009849020279943943, 0.007996265776455402, -0.03661010414361954, -0.037224914878606796, 0.05070336535573006, 0.0020950445905327797, 0.022110197693109512, -0.0361529104411602, -0.009727866388857365, 0.05169554427266121, -0.11978545039892197, 0.0029700954910367727, 0.0013352653477340937, -0.02044760249555111, -0.06619369983673096, 4.4500029616756365e-05, 0.012418258935213089, -0.002424633828923106, 0.0006805554730817676, 0.002168100792914629, 0.0199749656021595, -0.0039910380728542805, -0.004903365857899189, -5.402691749623045e-05, -0.019818568602204323, -0.00010459467011969537, -0.016151081770658493, -0.002360037760809064, 0.012175867334008217, -0.0013137271162122488, 0.039258550852537155, -0.02522825449705124, -0.0004457157920114696, 0.03697161003947258, 0.021865975111722946, -0.007757684215903282, 0.019068248569965363, -0.15842002630233765, 0.000158625582116656, -0.014343997463583946, 0.015526161529123783, -0.09605729579925537, -0.007691359147429466, -0.024424510076642036, 0.019963083788752556, -0.0018876208923757076, 0.02422427013516426, 0.0010253143263980746, -0.004927531816065311, -0.005871191620826721, -0.061526596546173096, -0.011267262510955334, -0.005279363598674536, 0.0072090462781488895, 0.017589624971151352, -0.0235123448073864, -0.003032931126654148, -0.018489953130483627, 0.030763287097215652, 0.07409337162971497, 0.06068030372262001, 0.00028255998040549457, -0.008609925396740437, 0.015640098601579666, 0.0028825232293456793, -0.022870294749736786, -0.017583360895514488, -0.025228051468729973, 0.0703650638461113, 0.010575173422694206, 0.000508734374307096, 0.00024253061565104872, -0.05002463608980179, -0.08455971628427505, -0.021648507565259933, 0.0075172120705246925, -0.011179197579622269, -0.140869140625, -0.07680065929889679, 0.03611060231924057, 0.007108293008059263, 0.007358755450695753, -0.03994733467698097, 6.048684736015275e-05, -0.016770686954259872, 0.018721865490078926, -0.06712035834789276, -0.0031442928593605757, 0.020713692530989647, -0.040873412042856216, -0.0009463416063226759, -0.05574683099985123, -0.026284988969564438, 0.12969666719436646, -0.008756346069276333, -0.0016717483522370458, -0.0022594977635890245, -0.0002962453872896731, 0.05431677773594856, -0.05527939274907112, -0.013866720721125603, -0.007185205817222595, 0.002322680316865444, -0.020932167768478394, 0.0441780611872673, -0.011690742336213589, -0.02023707889020443, 0.03150682896375656, 0.01717507652938366, -0.002360881771892309, 0.01335529237985611, 0.01374414935708046, -8.521639392711222e-05, 0.019863862544298172, -0.007387309335172176, 0.015519806183874607, 0.035218965262174606, -0.0034483144991099834, 0.014220152050256729, -0.0038606508169323206, 0.009912952780723572, 0.006749975960701704, 0.020986028015613556, -0.018002908676862717, 0.0068858591839671135, 0.09610975533723831, -0.012608454562723637, -0.014120953157544136, 0.003923949785530567, 0.0006516794092021883, 0.002025855937972665, -0.015165524557232857, -0.010590368881821632, 0.011101128533482552, 0.009055805392563343, 0.027430256828665733, -0.007941021583974361, -0.020289573818445206, -0.0016415599966421723, -0.015334050171077251, 0.04038785398006439, 0.06659717112779617, 0.009728134609758854, -0.010226675309240818, 0.09206598252058029, -0.011820762418210506, 0.15687096118927002, 0.011942191980779171, -0.01279282383620739, -0.018107987940311432, -0.014913886785507202, 0.02095501311123371, 0.09654147922992706, 0.027789248153567314, 0.005855438765138388, -0.006108059082180262, 0.005201911553740501, -0.04893317446112633, 0.03349804878234863, 0.048827629536390305, 0.0020576047245413065, 0.08556235581636429, -0.00445112818852067, -0.02524373307824135, 0.004902684595435858, -0.022483505308628082, -0.05688001960515976, -0.001692972844466567, 0.0091233029961586, 0.007418845314532518, -0.0655640959739685, 0.00241093710064888, 0.0014920602552592754, 0.0425410270690918, -0.004331882111728191, 0.0041695586405694485, 0.0074159689247608185, -0.010286101140081882, 0.0008421321399509907, -0.01507796160876751, -0.02645290642976761, 0.034613098949193954, 0.002001713728532195, -0.02450665645301342, 0.0642310231924057, -0.003566940315067768, 0.010413526557385921, 0.0019307609181851149, 0.009103386662900448, -0.022972041741013527, 0.03295522555708885, 0.021695394068956375, 0.03785400465130806, 0.046094443649053574, 0.000802470080088824, 0.07230087369680405, 0.1621496081352234, -0.01710992120206356, 0.017116360366344452, -0.003968927077949047, -0.0014961151173338294, 0.02052491530776024, -0.08758462220430374, 0.0015600310871377587, -0.046777259558439255, 0.007386361248791218, -0.09306277334690094, 0.04426100105047226, -0.0011965336743742228, -0.029857831075787544, 0.020855538547039032, 0.006058474536985159, 0.011503287591040134, -0.07050139456987381, -0.045740995556116104, 0.06833469122648239, -0.03364093229174614, -0.05108878016471863, 0.0010832770494744182, 0.051329754292964935, -0.02047533541917801, -0.024332504719495773, 0.04710102453827858, 0.005642229225486517, -0.17120857536792755, -0.053974322974681854, -0.00024229768314398825, -0.037303626537323, 0.05357522889971733, 0.04519163444638252, 0.023959675803780556, -0.08916816860437393, 0.04004434868693352, 0.02822493575513363, -0.005961699411273003, -0.0031027470249682665, 0.009038782678544521, 0.12208981812000275, 0.022689219564199448, 0.05210926756262779, 0.04358935356140137, 0.05395086482167244, -0.07030433416366577, 0.010327277705073357, 0.04220324754714966, -0.006152571644634008, -0.0042505101300776005, 0.06001206114888191, 0.03754567354917526, 0.0705622211098671, 0.10149948298931122, 0.0347297266125679, -0.02889432944357395, 0.01737780123949051, -0.016513237729668617, 0.018678022548556328, -0.034577954560518265, 0.006732461974024773, -0.03791366145014763, -0.028777560219168663, 0.05453009158372879, 0.05868089571595192, 0.0209988821297884, -0.020423555746674538, 0.024084068834781647, 0.02066599577665329, -0.008737285621464252, 0.049835801124572754, 0.01018644217401743, 0.02247137948870659, -0.07440292090177536, 0.0009008999331854284, 0.008444479666650295, -0.01927870325744152, -0.001480615115724504, 0.010956154204905033, 0.0041903164237737656, -0.021637363359332085, -0.003222318133339286, 0.0006043090834282339, 0.009090390056371689, 0.017297448590397835, -0.016761910170316696, -0.004439832177013159, 0.0023721554316580296, -0.006629355251789093, -0.01577703468501568, -0.02231396734714508, -0.012147280387580395, 0.024384507909417152, 0.0734434723854065, -0.006362995598465204, 0.04000701382756233, -0.06355106830596924, 0.00036507254117168486, -0.0019215958891436458, 0.006080011837184429, -0.01784593053162098, 0.0028938502073287964, 0.03488690406084061, 0.0033611436374485493, -0.005936960224062204, -0.015721792355179787, 0.0005704420618712902, 0.026121199131011963, -0.0053590950556099415, 0.0015031154034659266, 0.0013255857629701495, -0.004461848642677069, -0.003956012427806854, -0.015391852706670761, -0.12603476643562317, -0.018726540729403496, 0.030226251110434532, 0.007380446419119835, 0.052882175892591476, 0.011770013719797134, 0.0006578626926057041, 0.01858638972043991, 0.018677527084946632, 0.006895923055708408, -0.10449162125587463, -0.007623111363500357, -0.12486700713634491, 0.025811683386564255, -0.02108779177069664, -0.00017191731603816152, 0.0013066722312942147, -0.029806241393089294, -0.06973996013402939, -0.08078141510486603, 0.0010088221170008183, -0.006372619420289993, -0.023523222655057907, -0.11841124296188354, 0.10991694033145905, -0.004744792822748423, 0.006372462958097458, -0.02367105521261692, 0.0006723027909174562, 0.0020659922156482935, 0.0043139080516994, 0.04901353642344475, -0.007178493309766054, 0.04362502694129944, -0.033871252089738846, 0.0002708529063966125, 0.001276012510061264, 0.008879032917320728, 0.10033253580331802, -0.04005378857254982, -0.009315122850239277, 0.0006836183019913733, -0.017640985548496246, 0.08471003919839859, -0.12533991038799286, -0.00324145145714283, 0.0008491971530020237, -0.019550297409296036, 0.005379296839237213, -0.016513166949152946, 0.02078126184642315, -0.02006508968770504, 0.042964424937963486, 0.024721521884202957, -0.0006602638750337064, 0.0004384092171676457, -0.001481348997913301, -0.00954090990126133, -0.09274710714817047, -0.024158796295523643, -0.014115487225353718, -0.00420549139380455, -0.006012787111103535, 0.006663432344794273, -0.0003096313448622823, -0.052934370934963226, -0.0007867715321481228, -0.0020756397861987352, 0.006626776419579983, -0.008046253584325314, 0.08914285153150558, -0.014128541573882103, 0.016330331563949585, -0.05484968051314354, -0.005270944442600012, -0.005115148611366749, -0.024328775703907013, -0.009850037284195423, 0.050707750022411346, 2.76247246802086e-05, 0.005297381430864334, -0.03549877181649208, -0.007844273932278156, -0.0004477986949495971, -0.012675600126385689, 0.0055258446373045444, 0.03601697087287903, 0.010670811869204044, -0.013754385523498058, 0.010317996144294739, 0.12176620215177536, 0.14247429370880127, 0.03119608946144581, -0.0063134776428341866, 0.010365165770053864, 0.005065322387963533, 0.0200467798858881, -0.009755789302289486, 0.013181934133172035, -0.0037314030341804028, 0.011809553019702435, -0.06588097661733627, 0.011867423541843891, 0.03254108875989914, 0.1072952151298523, -0.0023633097298443317, 0.018777722492814064, -0.011542984284460545, -0.07634368538856506, 0.010750235989689827, -0.01840910315513611, 0.05822649970650673, 0.0006635244353674352, 0.020310647785663605, 0.019721295684576035, -0.009749611839652061, -0.021940920501947403, 0.002765238517895341, -0.010774736292660236, -0.008350850082933903, -0.04276933893561363, 0.0024991710670292377, -0.06090538203716278, -0.019623873755335808, 0.0029114035423845053, 0.013588272035121918, -0.0037953038699924946, -0.07507818937301636, -0.019919240847229958, 0.039668090641498566, 0.010285226628184319, 0.0075608198530972, -0.020331280305981636, -0.003767352784052491, 0.04762320965528488, 0.015044720843434334, 0.05955404043197632, 0.04873934015631676, 0.011920808814466, 0.006765647325664759, 0.07442411780357361, 0.182805135846138, 0.00824466161429882, 0.007940405048429966, 0.02526419423520565, 0.0006638519698753953, 0.032381221652030945, 0.0022450091782957315, -0.022812549024820328
	};
#pragma HLS ARRAY_PARTITION variable=bias type=complete
#pragma HLS BIND_STORAGE variable=weights impl=lutram type=ram_s2p

    float sum = 0;
    int in_offset = 0;
    int weight_offset = 0;
    int data_offset = 0;

	// Convolution layer
    ConvFilter:
	for (int filter = 0; filter < FC2_3_DENSE_SIZE; filter++) {
		ConvY:
		for (int i = 0; i < FC1_DENSE_SIZE ;i++) {
#pragma HLS PIPELINE off=false ii=7
			weight_offset = filter * FC1_DENSE_SIZE + i;
			output[filter] += input[i] * weights[weight_offset];

		}
//		output[filter] = sum + bias[filter];
//		sum = 0;
	}

	for(int i = 0 ; i < FC2_3_DENSE_SIZE; i++){
		output[i] += bias[i];
	}

	return;
	
}


void onet_accel(float* input, float* conv_mp_2_weights, float* conv_mp_3_weights, float* conv_4_weights, float* dense_1_weights, float* output1, float* output2, float* output3){
// #pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=INPUT_SIZE * INPUT_SIZE * CONV1_IN_CHANNEL
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=CONV1_OUT_SIZE * CONV1_OUT_SIZE * CONV1_FILTER
#pragma HLS INTERFACE m_axi port=conv_mp_2_weights offset=slave bundle=gmem1 depth=CONV2_SIZE * CONV2_SIZE * CONV2_FILTER * CONV2_IN_CHANNEL
#pragma HLS INTERFACE m_axi port=conv_mp_3_weights offset=slave bundle=gmem2 depth=CONV3_SIZE * CONV3_SIZE * CONV3_FILTER * CONV3_IN_CHANNEL
#pragma HLS INTERFACE m_axi port=conv_4_weights offset=slave bundle=gmem3 depth=CONV4_SIZE * CONV4_SIZE * CONV4_FILTER * CONV4_IN_CHANNEL
#pragma HLS INTERFACE m_axi port=dense_1_weights offset=slave bundle=gmem4 depth=FC1_DENSE_SIZE * CONV4_OUT_SIZE * CONV4_OUT_SIZE * CONV4_FILTER
#pragma HLS INTERFACE m_axi port=output1 offset=slave bundle=gmem5 depth = FC2_1_DENSE_SIZE
#pragma HLS INTERFACE m_axi port=output2 offset=slave bundle=gmem5 depth = FC2_2_DENSE_SIZE
#pragma HLS INTERFACE m_axi port=output3 offset=slave bundle=gmem5 depth = FC2_3_DENSE_SIZE
#pragma HLS INTERFACE s_axilite port=return
	float out1[POOL1_OUT_SIZE * POOL1_OUT_SIZE * CONV1_FILTER] = {0};
	float out2[POOL2_OUT_SIZE * POOL2_OUT_SIZE * CONV2_FILTER] = {0};
	float out3[CONV3_OUT_SIZE * CONV3_OUT_SIZE * CONV3_FILTER] = {0};
	float out4[CONV4_OUT_SIZE * CONV4_OUT_SIZE * CONV4_FILTER] = {0};
	float flatten_out[CONV4_OUT_SIZE * CONV4_OUT_SIZE * CONV4_FILTER] = {0};
	float out5[FC1_DENSE_SIZE] = {0};

	mp_1_accel(input, /*weights,*/ out1);
	conv_mp_2_accel(out1, conv_mp_2_weights, out2);
	conv_mp_3_accel(out2, conv_mp_3_weights, out3);
	conv_4_accel(out3, conv_4_weights, out4);
	flatten_accel(out4, flatten_out);
	dense_1_accel(flatten_out, dense_1_weights, out5);
	dense_2_1_accel(out5, /*weight,*/ output1);
	dense_2_2_accel(out5, /*weight,*/ output2);
	dense_2_3_accel(out5, output3);

};
