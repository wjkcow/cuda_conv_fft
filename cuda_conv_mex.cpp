#include <mex.h>				// Mex header
#include <stdio.h>
#include <math.h>
#include <string.h>
//
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Include CUDA runtime and CUFFT
#include "cuda_conv.h"

#define	IMAGE   	prhs[0]		// The stack of images you want to convolve: n x m x c matrix of single precision floating point numbers.
#define KERNEL      prhs[1]	// The kernel: i x j matrix of single precision floating point numbers.
#define	OUTPUT   	plhs[0]		// Convolved stack of images. If in valid mode, this will be (n-i+1) x (m-j+1) x c  matrix of single ...


void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
	unsigned img_x, img_y, img_s, img_dim, ker_dim, ch, ker_s;
	float *img, *ker, *result;
	int resultdim[4];
	mexLock();
	if (nrhs != 2) {
		mexErrMsgTxt("Three input arguments required.");
	} else if (nlhs > 1) {
		mexErrMsgTxt("Too many output arguments.");
	}

	const mwSize *imagedims;
	const mwSize *kerdims;

 	img_dim = mxGetNumberOfDimensions(IMAGE);
	imagedims = mxGetDimensions(IMAGE);

	ker_dim = mxGetNumberOfDimensions(KERNEL);
	kerdims = mxGetDimensions(KERNEL);
	if (img_dim != 4)
		mexErrMsgTxt("Only support 4-D image");
	if (ker_dim != 4)
		mexErrMsgTxt("Only support 4-D kernel");
	img_x = (unsigned)imagedims[0];
	img_y = (unsigned)imagedims[1];
	ch    = (unsigned)imagedims[2];
	img_s = (unsigned)imagedims[3];


	if((unsigned)kerdims[0] != img_x)
		mexErrMsgTxt("ker_x should equals img_x");
	if((unsigned)kerdims[1] != img_y)
		mexErrMsgTxt("ker_y should equals img_y");
	if ((unsigned)kerdims[2] != ch)
		mexErrMsgTxt("ch should match");
	ker_s = (unsigned)kerdims[3];

	img = (float *) mxGetData(IMAGE);
	ker = (float *) mxGetData(KERNEL);

	resultdim[0] = (mwSize)img_x;
	resultdim[1] = (mwSize)img_y;
	resultdim[2] = (mwSize)ker_s;
	resultdim[3] = (mwSize)img_s;
	OUTPUT = mxCreateNumericArray(4, resultdim, mxSINGLE_CLASS, mxREAL);
	if (OUTPUT == NULL)
			mexErrMsgTxt("Could not allocate output array");
	result = (float *) mxGetData(OUTPUT);
	mGpu_conv(img, filter, cpuAns,
		img_x, ker_s, ch, img_s, ker_s);
	return;

}
