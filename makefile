all: cuda_conv.mexa64


test: cuda_fft_lib.o main.cpp
	nvcc -Xcompiler -fopenmp -lgomp -o program -L/usr/local/cuda/lib64 -lcuda -lcudart -lcufft main.cpp  cuda_fft_lib.cu -O4


program: cuda_fft_lib.o main.cpp
	nvcc -o program -L/usr/local/cuda/lib64 -lcuda -lcudart -lcufft main.cpp  cuda_fft_lib.o 

cuda_fft_lib.o: cuda_fft_lib.cu
	nvcc -c -arch=sm_35 -Xcompiler -fPIC -c -o cuda_fft_lib.o cuda_fft_lib.cu 

clean:
	rm -rf *o program

cuda_conv.o: cuda_conv.c
    mex -c -o cuda_conv.o cuda_conv.c

mx.o: cuda_conv.o cuda_fft_lib.o
	nvcc -arch=sm_35 -Xcompiler -fPIC -fopenmp -lgomp -o program -L/usr/local/cuda/lib64 -o mx.o -dlink cuda_fft_lib.o cuda_conv.o -lcudadevrt

cuda_conv.mexa64: mx.o
    mex -o cuda_conv.mexa64 mx.o -L/usr/local/cuda/lib64 -lcudart -lcudadevrt
