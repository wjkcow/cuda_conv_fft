#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>
#include "cuda_fft_lib.h"
#include "cuda_conv.h"


bool mGpu_conv(float * cpuImg, float * cpuFilter, float *ans,
	int x, int y, int ch, int batch, int f_batch){
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);

	int gpuBatch = f_batch / num_gpus;
	int lastBatch = f_batch - gpuBatch * (num_gpus - 1);
	omp_set_num_threads(num_gpus);
	fprintf(stderr,"begin ");

	#pragma omp parallel
	{
		unsigned int cpu_thread_id = omp_get_thread_num();
		cuda_init(cpu_thread_id);
		if(cpu_thread_id == num_gpus - 1){
			conv_cufft(cpuImg, cpuFilter + (size_t)(cpu_thread_id )*
				x*y*ch*gpuBatch, ans + (size_t)(cpu_thread_id)* batch *x*y ,
				x, y, ch, batch, lastBatch);
		} else{
			conv_cufft(cpuImg, cpuFilter + (size_t)(cpu_thread_id )*
				x*y*ch*gpuBatch, ans + (size_t)(cpu_thread_id)* batch *x*y ,
				x, y, ch, batch, gpuBatch);
		}


	}
	fprintf(stderr,"DONE ");
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

		batch_ifft2(gpuProd, gpuAns , x, y, batch);
		my_cpu_ans = my_cpu_ans + x * y * batch;
		fprintf(stderr,"copying to %p",my_cpu_ans);
		checkCudaErrors(cudaMemcpy(my_cpu_ans, gpuAns, x * y * batch* sizeof(float), cudaMemcpyDeviceToHost));
	}

	checkCudaErrors(cudaFree(gpuProd));
	checkCudaErrors(cudaFree(gpuImg));
	checkCudaErrors(cudaFree(gpuFilter));
	checkCudaErrors(cudaFree(gpuAns));
	checkCudaErrors(cudaFree(gpuImg_C));
	checkCudaErrors(cudaFree(gpuFilter_C));
	return true;
}
