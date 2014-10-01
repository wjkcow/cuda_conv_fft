all: program

test: cuda_fft_lib.o main.cpp
	nvcc -Xcompiler -fopenmp -lgomp -o program -L/usr/local/cuda/lib64 -lcuda -lcudart -lcufft main.cpp  cuda_fft_lib.cu -O4


program: cuda_fft_lib.o main.cpp
	nvcc -o program -L/usr/local/cuda/lib64 -lcuda -lcudart -lcufft main.cpp  cuda_fft_lib.o 

cuda_fft_lib.o: cuda_fft_lib.cu
	nvcc -c -arch=sm_20 cuda_fft_lib.cu 

clean:
	rm -rf *o program
