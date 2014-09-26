#ifndef HELPER_H
#define HELPER_H


#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset();
#endif

#ifdef __CUDACC__
typedef float2 fComplex;
#else
typedef struct
{
	float x;
	float y;
} fComplex;
#endif

template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
			file, line, static_cast<unsigned int>(result), "error" , func);
		DEVICE_RESET
        // Make sure we call CUDA Device Reset before exiting
		exit(EXIT_FAILURE);
	}
}
#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )

#endif /* HELPER_H */