#ifndef CUDA_CONV_H
#define CUDA_CONV_H


bool mGpu_conv(float * cpuImg, float * cpuFilter, float *ans,
	int x, int y, int ch, int batch, int f_batch);

bool conv_cufft(float * cpuImg, float * cpuFilter, float *ans,
	int x, int y, int ch, int batch, int f_batch);

#endif /* CUDA_CONV_Hs */
