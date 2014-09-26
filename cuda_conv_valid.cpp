#include <mex.h>				// Mex header
#include <stdio.h>
#include <math.h>
#include <string.h>
#define	IMAGE   	prhs[0]		// The stack of images you want to convolve: n x m x c matrix of single precision floating point numbers.
#define KERNEL      prhs[1]	// The kernel: i x j matrix of single precision floating point numbers.
#define	OUTPUT   	plhs[0]		// Convolved stack of images. If in valid mode, this will be (n-i+1) x (m-j+1) x c  matrix of single ...


void mexFunction(int nlhs, mxArray * plhs[], int nrhs, const mxArray * prhs[])
{
	int img_x, img_y, img_s, ker_x, ker_y, ker_s, ch, img_dim, ker_dim;
	float *img, *ker, *result;
	int resultdim[4];

	if (nrhs != 3) {
		mexErrMsgTxt("Three input arguments required.");
	} else if (nlhs > 1) {
		mexErrMsgTxt("Too many output arguments.");
	}

	const mwSize *imagedims;
	const mwSize *kerdims;

 	img_dim = mxGetNumberOfDimensions(IMAGE);
	imagedims = (mwSize *) mxCalloc(img_dim, sizeof(mwSize));
	imagedims = mxGetDimensions(IMAGE);

	ker_dim = mxGetNumberOfDimensions(KERNEL);
	kerdims = (mwSize *) mxCalloc(ker_dim, sizeof(mwSize));
	kerdims = mxGetDimensions(ker_dim);
	if (img_dim != 4)
		mexErrMsgTxt("Only support 4-D image");
	if (img_dim != 4)
		mexErrMsgTxt("Only support 4-D kernel");
	img_x = imagedims[0];
	img_y = imagedims[1];
	ch = imagedims[2];
	img_s = imagedims[3];


	ker_x = kerdims[0];
	ker_y = kerdims[1];
	if (ch != kerdims[2])
		mexErrMsgTxt("Img ch should equals kernel ch");
	ker_s = kerdims[3];


	img = (float *) mxGetData(IMAGE);
	ker = (float *) mxGetData(KERNEL);

	resultdim[0] = img_x - ker_x + 1;
	resultdim[1] = img_y - ker_y + 1;
	resultdim[2] = ker_s;
	resultdim[3] = img_s;
	OUTPUT = mxCreateNumericArray(4, resultdim, mxSINGLE_CLASS, mxREAL);
	if (OUTPUT == NULL)
			mexErrMsgTxt("Could not allocate output array");
	*result = (float *) mxGetData(OUTPUT);

	if(af_conv(img, ker, img_x, img_y, img_s, ker_x, ker_y, ker_s, ch, result)){
		mexErrMsgTxt("Only support 4-D image");
    }

	return;

}
