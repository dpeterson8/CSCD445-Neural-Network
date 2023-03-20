#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "gpuNetwork.h"

__device__ float gpuSigmoid(float x) {
    return 1 / (1 + exp(-x));
}

__global__ void gpuFillInputLayer(float * input, float * inputLayer, int inputSize, int position) {
    int ix = (blockDim.x * blockIdx.x) + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = (iy * (gridDim.x * blockDim.x)) + ix;

    inputLayer[ix] = input[ix + inputSize * position];
    
    
}

__global__ void gpuActivateLayers( float * layers, float * nextLayer, float * nextWeights, float * nextBias, int inputSize, int nextSize) {

    if(threadIdx.x == 0) {
        nextLayer[blockIdx.x] = 0;

        nextLayer[blockIdx.x] += nextBias[blockIdx.x];
    }

    __syncthreads();

    nextLayer[blockIdx.x] += nextWeights[threadIdx.x + blockDim.x * blockIdx.x] * layers[threadIdx.x];

    __syncthreads();

    if(threadIdx.x == 0) {
        nextLayer[blockIdx.x] = gpuSigmoid(nextLayer[blockIdx.x]);
    }

    __syncthreads();
}
