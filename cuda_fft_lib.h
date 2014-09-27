#ifndef CUDA_FFT_LIB_H
#define CUDA_FFT_LIB_H

#include "helper.h"
extern "C" void cuda_init();
extern "C" bool conv_cufft(float * cpuImg, float * cpuFilter, float *ans,
	int x, int y, int ch, int batch, int f_batch);
extern "C" fComplex* batch_fft2(float * gpuIn, int x, int y, int batch);
extern "C" bool multi_sum_norm(fComplex * img, fComplex * filter, fComplex * result,
	int xy, int ch, int img_batch);
extern "C" bool batch_ifft2(fComplex * gpuIn, float * gpuOut, int x, int y, int batch);

#endif /* CUDA_FFT_LIB_H */