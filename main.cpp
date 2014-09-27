#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>
#include "helper.h"
#include "cuda_fft_lib.h"
//tes of conv
void test3(){
	fComplex *gpuImg_C, *gpuFilter_C, * gpuProd;
	float *gpuImg, * gpuFilter;
	float *gpuAns;
	int ch = 3;
	int xy = 9;
	int stack  = 2;
	float * img = (float *) malloc(9 * 3 *2* sizeof(float));
	float * filter = (float *) malloc(9 * 3 * sizeof(float));
	float * cpuAns = (float *) malloc(9 * 2 * sizeof(float));

	float * gpuResult;
	for (int i = 0; i < 9 * 3 *2; ++i){
		img[i] = i;
	}
	memset(filter, 0, 9*3 * sizeof(float));
	fiter[0] = 1;
	fiter[1] = 1;
	fiter[3] = 1;
	fiter[4] = 1;

	fiter[9] = 1;
	fiter[10] = 1;
	fiter[12] = 1;
	fiter[13] = 1;

	fiter[18] = 1;
	fiter[19] = 1;
	fiter[21] = 1;
	fiter[22] = 1;

	conv_cufft(img, filter, cpuAns,
		3, 3, 3, 2, 1);
	for(int s = 0; s < 2 ; ++s){
		for (int i = 0; i < 3; ++i)
		{
			for(int j = 0; j < 3; ++J){
				fprintf(stderr, "%f \n",cpuAns[s * 9 i*3 + j] ); 
			}
			fprintf(stderr, "***********" ); 
		}
	}




}
void test2(){
	fComplex * cpuIn1, *cpuIn2, *gpuIn, *gpuIn2, *gpuOut,* cpuOut;
	cpuIn1 = (fComplex *)malloc(  96*96*3*5000 * sizeof(fComplex));
	cpuIn2 = (fComplex *)malloc(  96*96*3 * sizeof(fComplex));
	cpuOut = (fComplex *)malloc(  96*96*5000 * sizeof(fComplex));

	cuda_init();
	checkCudaErrors(cudaMalloc((void **)&gpuIn,  96*96*3 *5000 * sizeof(fComplex)));
	checkCudaErrors(cudaMalloc((void **)&gpuIn2,  96*96*3 * sizeof(fComplex)));
	checkCudaErrors(cudaMalloc((void **)&gpuOut,  96*96*5000 * sizeof(fComplex)));
	checkCudaErrors(cudaMemset(gpuOut, 1, 96*96*5000 * sizeof(fComplex)));

	checkCudaErrors(cudaMemcpy(gpuIn, cpuIn1, 96*96*5000 * sizeof(fComplex), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpuIn2, cpuIn2,  96*96*3  * sizeof(fComplex), cudaMemcpyHostToDevice));

	multi_sum_norm(gpuIn, gpuIn2, gpuOut,96*96, 3, 5000);
	checkCudaErrors(cudaMemcpy(cpuOut, gpuOut,  12 * sizeof(fComplex), cudaMemcpyDeviceToHost));


	for (int i = 0; i < 12; ++i)
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
	test3();

	return 0;
}
