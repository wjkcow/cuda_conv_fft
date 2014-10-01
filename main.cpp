#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
// Include CUDA runtime and CUFFT
#include <cuda_runtime.h>
#include <cufft.h>
#include "helper.h"
#include "cuda_fft_lib.h"
//tes of conv

void test4(){


	fComplex *gpuImg_C, *gpuFilter_C, * gpuProd;
	float *gpuImg, * gpuFilter;
	float *gpuAns;
	int ch = 3;
	int xy = 9;
	int stack  = 2;
	float * img = (float *) malloc((size_t)96 * 96 * 3 * 5000* sizeof(float));
	float * filter = (float *) malloc((size_t)96 * 96 *3 * 96 *sizeof(float));
	float * cpuAns = (float *) malloc((size_t)96 *96*5000 *96  * sizeof(float));
	int num_gpus = 0;
	cudaGetDeviceCount(&num_gpus);
	fprintf(stderr,"begin\n");
	omp_set_num_threads(num_gpus);
	#pragma omp parallel
	{
	unsigned int cpu_thread_id = omp_get_thread_num();
	if(cpu_thread_id == 0){
		cuda_init(0);
		conv_cufft(img, filter, cpuAns,
		96, 96, 3, 5000, 24);
	} else {
		cuda_init(cpu_thread_id);
		conv_cufft(img, filter + 96*96*3*30 +(cpu_thread_id - 1)*
		96*96*3*22,cpuAns + (size_t)(cpu_thread_id-1) *5000 *96*96 ,
		
		96,96,3,5000,24);

	}
	}
	fprintf(stderr,"AllDone");
}
void test3(){
	fComplex *gpuImg_C, *gpuFilter_C, * gpuProd;
	float *gpuImg, * gpuFilter;
	float *gpuAns;
	int ch = 3;
	int xy = 9;
	int stack  = 2;
	float * img = (float *) malloc(96 * 96 * 3 * 5000* sizeof(float));
	float * filter = (float *) malloc(96 * 96 *3 * 96 *sizeof(float));
	float * cpuAns = (float *) malloc(96 *96*5000 *96  * sizeof(float));

	float * gpuResult;
	for (int i = 0; i < 9 * 3 *2; ++i){
		img[i] = i;
		if(i >=27){
			img[i] = i - 27;
		}
	}
	memset(filter, 0, 9*3 * sizeof(float));
	filter[0] = 1;
	filter[1] = 1;
	filter[3] = 1;
	filter[4] = 1;

	filter[9] = 1;
	filter[10] = 1;
	filter[12] = 1;
	filter[13] = 1;

	filter[18] = 1;
	filter[19] = 1;
	filter[21] = 1;
	filter[22] = 1;

	conv_cufft(img, filter, cpuAns,
		96, 96, 3, 5000, 1);
	for(int s = 0; s < 2 ; ++s){
		for (int i = 0; i < 3; ++i)
		{
			for(int j = 0; j < 3; ++ j){
				fprintf(stderr, "%f  ",cpuAns[s * 9 + i*3 + j] );
			}
			fprintf(stderr, " \n" );

		}
		fprintf(stderr, "***********" );
	}




}
void test2(){
	fComplex * cpuIn1, *cpuIn2, *gpuIn, *gpuIn2, *gpuOut,* cpuOut;
	cpuIn1 = (fComplex *)malloc(  96*96*3*5000 * sizeof(fComplex));
	cpuIn2 = (fComplex *)malloc(  96*96*3 * sizeof(fComplex));
	cpuOut = (fComplex *)malloc(  96*96*5000 * sizeof(fComplex));

	cuda_init(0);
	checkCudaErrors(cudaMalloc((void **)&gpuIn,  96*96*3 *5000 * sizeof(fComplex)));
	checkCudaErrors(cudaMalloc((void **)&gpuIn2,  96*96*3 * sizeof(fComplex)));
	checkCudaErrors(cudaMalloc((void **)&gpuOut,  96*96*5000 * sizeof(fComplex)));
	checkCudaErrors(cudaMemset(gpuOut, 1, 96*96*5000 * sizeof(fComplex)));

	checkCudaErrors(cudaMemcpy(gpuIn, cpuIn1, 96*96*5000 * sizeof(fComplex), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpuIn2, cpuIn2,  96*96*3  * sizeof(fComplex), cudaMemcpyHostToDevice));

	multi_sum_norm(gpuIn, gpuIn2, gpuOut,96*96, 3, 5000, (float)96*(96/2 + 1));
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
	float input[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
	float *gpuIn;
	fComplex *gpuOut;
	fComplex cpuOut[12];

	cuda_init(0);

	checkCudaErrors(cudaMalloc((void **)&gpuIn,  18 * sizeof(float)));
	checkCudaErrors(cudaMemcpy(gpuIn, input, 18 * sizeof(float), cudaMemcpyHostToDevice));

	gpuOut = batch_fft2(gpuIn, 3, 3, 2 );
	

	checkCudaErrors(cudaMemcpy(cpuOut, gpuOut, 12 * sizeof(fComplex), cudaMemcpyDeviceToHost));

	batch_ifft2(gpuOut, gpuIn, 3,3,2);

	checkCudaErrors(cudaMemcpy(input, gpuIn, 18 * sizeof(float), cudaMemcpyDeviceToHost));


	for (int i = 0; i < 12; ++i)
	{
		fprintf(stderr, "%f , %f \n",cpuOut[i].x, cpuOut[i].y); 
	}

	for (int i = 0; i < 18; ++i)
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
	test4();

	return 0;
}
