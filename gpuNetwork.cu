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


    inputLayer[ix] = input[ix + inputSize * position];
    // printf("input: %d ", input[ix + inputSize * position]);
    __syncthreads();
}

// __global__ void gpuActivateLayers( float * layers, float * nextLayer, float * nextWeights, float * nextBias, int inputSize, int nextSize) {
//     __shared__ float total;
//     int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

//     if(idx < nextSize) {
//         float result;

//         for(int i = 0; i < inputSize; i++) {
//             result += layers[i] * nextWeights[i + idx * inputSize];
//         }
        
//         nextLayer[idx] = gpuSigmoid(result);
//     }

//     __syncthreads();
// }

__global__ void gpuActivateLayers( float * layers, float * nextLayer, float * nextWeights, float * nextBias, int inputSize, int nextSize) {
    __shared__ float total;


    if(threadIdx.x == 0) {
        nextLayer[blockIdx.x] = 0;

        total = nextBias[blockIdx.x];
    }

    __syncthreads();

    float tempValue = (nextWeights[threadIdx.x + blockDim.x * blockIdx.x] * layers[threadIdx.x]);
    atomicAdd(&total, tempValue);
    // __sync_fetch_and_add(total, nextWeights[threadIdx.x + blockDim.x * blockIdx.x] * layers[threadIdx.x]);

    __syncthreads();

    if(threadIdx.x == 0) {

        nextLayer[blockIdx.x] = gpuSigmoid(total);

    }
    __syncthreads();

}

__global__ void gpuOutError(float * deltaOut, float * correctInput, float * outLayer, int outSize, int dataPosition) {
    int ix = (blockDim.x * blockIdx.x) + threadIdx.x;

    deltaOut[ix] = deltaOut[ix] * (1 - outLayer[ix]) * (outLayer[ix] - correctInput[ix + outSize * dataPosition]);
    
    __syncthreads();
}

__global__ void gpuHiddenError(float * deltaCurrent, float * deltaPrev, float * currentLayer, float * prevLayerWeight, int curSize, int prevSize) {
    int ix = (blockDim.x * blockIdx.x) + threadIdx.x;
    int thId = threadIdx.x;
    
    __shared__ float error;

    float solution = (prevLayerWeight[ix] * deltaPrev[thId]);
    atomicAdd(&error, solution);

    __syncthreads();

    if(thId == 0) {
        deltaCurrent[blockIdx.x] = error * currentLayer[blockIdx.x] * (1 - currentLayer[blockIdx.x]);
    }

}

__global__ void gpuBackProp(float * curBias, float * deltaCur, float * curWeights, float * prevLayer, float learnRate, int curSize, int prevSize) {
    int ix = (blockDim.x * blockIdx.x) + threadIdx.x;
    int trId = threadIdx.x; 
    // printf(" learn rate: %d ", curWeights[blockIdx.x]);
    if(trId == 0) {
        curBias[blockIdx.x] = curBias[blockIdx.x] - (0.2 * deltaCur[blockIdx.x]);
        // printf("Hello: ,%d ", ix);
    }
    
    curWeights[ix] = curWeights[ix] - (0.2 * deltaCur[blockIdx.x]) * prevLayer[trId];
    

}

__global__ void gpuBackPropTwo(float * curBias, float * deltaCur, float * curWeights, float * prevLayer, float learnRate, int curSize, int prevSize) {
    int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

    if(idx < curSize) {
        curBias[idx] = curBias[idx] - (learnRate * deltaCur[idx]);  

        for(int i = 0; i < prevSize; i++) {
            curWeights[i + idx * prevSize] += curWeights[i + idx * prevSize] - (learnRate * deltaCur[idx]) * prevLayer[i];
        }
    }

    __syncthreads();

}