CUDA_INSTALL_PATH := /usr/local/cuda
 
CUDA  := $(CUDA_INSTALL_PATH)
 
INC     := -I$(CUDA)/include 
LIB     := -L$(CUDA)/lib
 
# Mex script installed by Matlab
MEX = /usr/local/MATLAB/R2013b/bin/mex 

# Flags for the CUDA nvcc compiler
NVCCFLAGS :=  -O=4 -arch=sm_35 --ptxas-options=-v -m 64
#THIS IS FOR DEBUG !!! -g -G
# IMPORTANT : don't forget the CUDA runtime (-lcudart) !
LIBS     := -lcudart -lcusparse -lcublas
 
# Regular C++ part
CXX = g++
CFLAGS = -Wall -c -O2 -fPIC $(INC)
LFLAGS = -Wall
AR = ar

all: mex
 
kernels:
     nvcc cuda_conv.cpp cuda_fft_lib.cu -c -o cuda_fft_lib.cu.o $(INC) $(NVCCFLAGS) -lcufft

cuda_fft_lib.a:     kernels
     ${AR} -r libcuda_fft_lib.a cuda_fft_lib.cu.o
 
mex:     cuda_fft_lib.a
     ${MEX} -L. -lcuda_fft_lib -v cuda_conv_mex.cpp -L$(CUDA)/lib $(LIBS)
     install_name_tool -add_rpath /usr/local/cuda/lib cuda_conv.mexa64
 
clean:
     rm *.o a.out *.a *.mexa64* *~
