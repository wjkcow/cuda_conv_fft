#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>
#include "cuda_fft_lib.h"
#include "helper.h"


inline __device__ void mulAndScale(fComplex &a, fComplex &b, fComplex &t, float c)
{
    t.x += c *(a.x * b.x - a.y * b.y);
    t.y += c *(a.y * b.x + a.x * b.y);
}

// z = x .* y / c channel added together
__global__ void cuda_multi_add(fComplex * x_base, fComplex * y, fComplex * z_base,
	 int xy, int ch, float ratio){

	int img_stack      =  blockIdx.x;  // 1 - 5000

	fComplex *x = x_base + img_stack * xy * ch;
	fComplex *z = z_base + img_stack * xy;
	for(int c = 0; c < ch ; c++){
    	for(int i = 0; i < xy ; i ++){
    		mulAndScale(x[c * xy + i], y[c * xy + i], z[i], 1.0/ratio);
    	}
	}
}

extern "C" void cuda_init(int n){
	cudaSetDevice(n);
	DEVICE_RESET
}

extern "C" fComplex* batch_fft2(float * gpuIn, int x, int y, int batch){
	cufftHandle fftPlanFwd;
	fComplex *gpuOut;
	int dim[2];
	dim[0] = x;
	dim[1] = y;
	int fftSize = x * (y / 2 +1)* batch;
	if (cufftPlanMany(&fftPlanFwd, 2, dim, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, batch) != CUFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT Error: Unable to create plan\n"); 
		return 0;	
	}

	checkCudaErrors(cudaMalloc((void **)&gpuOut,  fftSize * sizeof(fComplex)));
	checkCudaErrors(cudaMemset(gpuOut, 0, fftSize * sizeof(fComplex)));
	checkCudaErrors(cufftExecR2C(fftPlanFwd, (cufftReal *)gpuIn, (cufftComplex *)gpuOut));
	checkCudaErrors(cudaDeviceSynchronize());
	cufftDestroy(fftPlanFwd);
	return gpuOut;
}
extern "C" bool multi_sum_norm(fComplex * img, fComplex * filter, fComplex * result,
	int xy, int ch, int img_batch, float ratio){

	dim3 grid(img_batch, 1);
	dim3 block(1);
	cuda_multi_add<<<grid,block,0,0>>>(img, filter, result, xy, ch, ratio);
	checkCudaErrors(cudaThreadSynchronize());
	return true;
}
extern "C" bool batch_ifft2(fComplex * gpuIn, float * gpuOut, int x, int y, int batch){
	cufftHandle fftPlanInv;
	int dim[2];
	dim[0] = x;
	dim[1] = y;
//	int fftSize = x * y * batch;
	if (cufftPlanMany(&fftPlanInv, 2, dim, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R,batch) != CUFFT_SUCCESS)
	{ 
		fprintf(stderr, "CUFFT Error: Unable to create plan\n"); 
		return 0;	
	}
	checkCudaErrors(cufftExecC2R(fftPlanInv, (cufftComplex *)gpuIn,(cufftReal *)gpuOut));
    checkCudaErrors(cudaDeviceSynchronize());
	cufftDestroy(fftPlanInv);
	return true;
}



extern "C" bool conv_cufft(float * cpuImg, float * cpuFilter, float *ans,
	int x, int y, int ch, int batch, int f_batch){

	float * gpuImg, * gpuFilter, * gpuAns;
	fComplex *gpuImg_C, *gpuFilter_C, *gpuProd;
	size_t img_size = (size_t) x * y * ch * batch;
	size_t filter_size = (size_t) x * y * ch * f_batch;
	size_t ans_size = (size_t) x * y * batch;
	size_t conv_size = (size_t) x * (y/2 + 1) * batch;
//
//	fComplex *cpuProd =(fComplex *) malloc(ans_size * sizeof(fComplex));
//
//	cuda_init();
	checkCudaErrors(cudaMalloc((void **)&gpuProd,  conv_size * sizeof(fComplex)));
	checkCudaErrors(cudaMalloc((void **)&gpuImg,  img_size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&gpuFilter,  filter_size * sizeof(float)));
	checkCudaErrors(cudaMalloc((void **)&gpuAns,  ans_size * sizeof(float)));

	checkCudaErrors(cudaMemcpy(gpuImg, cpuImg, img_size * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpuFilter, cpuFilter, filter_size * sizeof(float), cudaMemcpyHostToDevice));

	gpuImg_C = batch_fft2(gpuImg, x, y, ch * batch );
	gpuFilter_C = batch_fft2(gpuFilter, x, y, ch * f_batch );
	float * my_cpu_ans = ans;
	for(int b = 0; b < f_batch; ++ b){
		fprintf(stderr, "batch %d\n", b);
		checkCudaErrors(cudaMemset(gpuProd, 0, conv_size * sizeof(fComplex)));
		fComplex *myFilter = gpuFilter_C + b * x * (y / 2 + 1) * ch;
		multi_sum_norm(gpuImg_C, myFilter, gpuProd, x*(y / 2 + 1) , ch, batch, (float)x*y);
//		checkCudaErrors(cudaMemcpy(cpuProd, gpuProd, ans_size * sizeof(float), cudaMemcpyDeviceToHost));
//
	//	for(int j = 0; j < 18; ++j){
		//	fprintf(stderr, "%d %f, %f \n",j,cpuProd[j].x , cpuProd[j].y );
		//}

		batch_ifft2(gpuProd, gpuAns , x, y, batch);
		my_cpu_ans = my_cpu_ans + x * y * batch;
		fprintf(stderr,"copying to %p",my_cpu_ans);
		checkCudaErrors(cudaMemcpy(my_cpu_ans, gpuAns, x * y * batch* sizeof(float), cudaMemcpyDeviceToHost));
		fprintf(stderr, "copy end");
	}

	checkCudaErrors(cudaFree(gpuProd));
	checkCudaErrors(cudaFree(gpuImg));
	checkCudaErrors(cudaFree(gpuFilter));
	checkCudaErrors(cudaFree(gpuAns));
	checkCudaErrors(cudaFree(gpuImg_C));
	checkCudaErrors(cudaFree(gpuFilter_C));
	return true;
}






