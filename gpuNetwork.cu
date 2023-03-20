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
    // if(threadIdx.x)
    // printf("%f ", inputLayer[threadIdx.x]);
    
}

__global__ void gpuActivateLayers( float * layers, float * nextLayer, float * nextWeights, float * nextBias, int inputSize, int nextSize) {

    if(threadIdx.x == 0) {
        nextLayer[blockIdx.x] = 0;

        nextLayer[blockIdx.x] += nextBias[blockIdx.x];
    }

    __syncthreads();

    float tempValue = (nextWeights[threadIdx.x + blockDim.x * blockIdx.x] * layers[threadIdx.x]);
    atomicAdd(&nextLayer[blockIdx.x], tempValue);

    __syncthreads();

    if(threadIdx.x == 0) {
        nextLayer[blockIdx.x] = gpuSigmoid(nextLayer[blockIdx.x]);
    }
    __syncthreads();
}

__global__ void gpuOutError(float * deltaOut, float * correctInput, float * outLayer, int outSize, int dataPosition) {
    int ix = (blockDim.x * blockIdx.x) + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = (iy * (gridDim.x * blockDim.x)) + ix;

    deltaOut[ix] = deltaOut[ix] * (1 - outLayer[ix]) * (outLayer[ix] - correctInput[ix + outSize * dataPosition]);
    
    __syncthreads();
}

__global__ void gpuHiddenError(float * deltaCurrent, float * deltaPrev, float * currentLayer, float * prevLayerWeight, int curSize, int prevSize) {
    
    

    for(int i = 0; i < curSize; i++) {
        float error = 0.0;
        for(int j = 0; j < prevSize; j++) {
            error += (prevLayerWeight[j + i * prevSize] * deltaPrev[j]);
        }
        deltaCurrent[i] = error * currentLayer[i] * (1 - currentLayer[i]);
    }
}