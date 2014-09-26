#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>
#include "helper.h"
#include "cuda_fft_lib.h"
void test2(){
	fComplex * cpuIn1, *cpuIn2, *gpuIn, *gpuIn2, *gpuOut,* cpuOut;
	cpuIn1 = (fComplex *)malloc(  12 * sizeof(fComplex));
	cpuIn2 = (fComplex *)malloc(  12 * sizeof(fComplex));
	cpuOut = (fComplex *)malloc(  12 * sizeof(fComplex));
	cuda_init();

	for (int i = 0; i < 12; ++i)
	{
		cpuIn1[i].x = i;
		cpuIn1[i].y = i + 10;
		cpuIn2[i].x = i + 11;
		cpuIn2[i].y = 11 - i;
		cpuOut[i].x = 0;
		cpuOut[i].y = 0;

	}
	checkCudaErrors(cudaMemset(gpuOut, 0, 12* sizeof(fComplex)));
	checkCudaErrors(cudaMalloc((void **)&gpuIn,  12 * sizeof(fComplex)));
	checkCudaErrors(cudaMalloc((void **)&gpuIn2,  12 * sizeof(fComplex)));
	checkCudaErrors(cudaMalloc((void **)&gpuOut,  12* sizeof(fComplex)));

	checkCudaErrors(cudaMemcpy(gpuIn, cpuIn1, 12 * sizeof(fComplex), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpuIn2, cpuIn2,  12 * sizeof(fComplex), cudaMemcpyHostToDevice));

	multi_sum_norm(gpuIn, gpuIn2, gpuOut,2, 3, 2);
	checkCudaErrors(cudaMemcpy(cpuOut, gpuOut,  12 * sizeof(fComplex), cudaMemcpyDeviceToHost));


	for (int i = 0; i < 8; ++i)
	{
		fprintf(stderr, "*********"); 
		fprintf(stderr, "cpuin %f , %f \n",cpuIn1[i].x, cpuIn1[i].y); 
		fprintf(stderr, "cpuin2 %f , %f \n",cpuIn2[i].x, cpuIn2[i].y); 
		fprintf(stderr, "cpuout %f , %f \n",cpuOut[i].x, cpuOut[i].y); 

	}


}
void test1(){
	float input[] = {1,2,3,4,5,6,7,8};
	float *gpuIn;
	fComplex *gpuOut;
	fComplex cpuOut[8];

	cuda_init();

	checkCudaErrors(cudaMalloc((void **)&gpuIn,  8 * sizeof(float)));
	checkCudaErrors(cudaMemcpy(gpuIn, input, 8 * sizeof(float), cudaMemcpyHostToDevice));

	gpuOut = batch_fft2(gpuIn, 2, 2, 2 );
	

	checkCudaErrors(cudaMemcpy(cpuOut, gpuOut, 8 * sizeof(fComplex), cudaMemcpyDeviceToHost));

	batch_ifft2(gpuOut, gpuIn, 2,2,2);

	checkCudaErrors(cudaMemcpy(input, gpuIn, 8 * sizeof(float), cudaMemcpyDeviceToHost));


	for (int i = 0; i < 8; ++i)
	{
		fprintf(stderr, "%f , %f \n",cpuOut[i].x, cpuOut[i].y); 
	}

	for (int i = 0; i < 8; ++i)
	{
		fprintf(stderr, "%f  \n",input[i]); 
	}


	// if (cufftPlanMany(&fftPlanInv, 2, dim, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C,batch) != CUFFT_SUCCESS)
	// { 
	// 	fprintf(stderr, "CUFFT Error: Unable to create plan\n"); 
	// 	return 0;	
	// }
	// checkCudaErrors(cufftDestroy(fftPlanFwd));
	// cufftDestroy(fftPlanInv)

	checkCudaErrors(cudaFree(gpuIn));
	checkCudaErrors(cudaFree(gpuOut));
}
int main(int argc, char  **argvs)
{
	test1();

	return 0;
}
