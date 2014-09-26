all: program

program: cuda_fft_lib.o
	nvcc -o program -L/usr/local/cuda/lib64 -lcuda -lcudart -lcufft main.cpp  cuda_fft_lib.o

cuda_fft_lib.o:
	nvcc -c -arch=sm_20 cuda_fft_lib.cu 

clean:
	rm -rf *o program
