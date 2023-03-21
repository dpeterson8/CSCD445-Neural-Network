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
    __syncthreads();
}

__global__ void kOutError(float * deltaOut, float * correctInput, float * outLayer, int outSize) {
    int idx = threadIdx.x;
    for(int i = 0; i < outSize; i++) {
        deltaOut[i + (idx * outSize)] = outLayer[i + (idx * outSize)] * (1 - outLayer[i + (idx * outSize)]) * (outLayer[i + (idx * outSize)] - correctInput[i + (outSize * idx)]);
    }
}

__global__ void kActivate(float * layers, float * nextLayer, float * nextWeights, float * nextBias, int inSize, int nextSize) {
    int idx = threadIdx.x;

    for(int i = 0; i < nextSize; i++) {
        float result = nextBias[i];
        for(int j = 0; j < inSize; j++) {
            result += layers[j + (idx * inSize)] * nextWeights[j + (i * inSize)];
        }
    
        nextLayer[i + (idx * nextSize)] = gpuSigmoid(result);
    }

}

__global__ void kError(float * deltaCurrent, float * deltaPrev, float * currentLayer, float * prevLayerWeight, int curSize, int prevSize) {
    int idx = threadIdx.x;

        for(int i = 0; i < curSize; i++) {
        float error = 0.0;
        for(int j = 0; j < prevSize; j++) {
            error += (prevLayerWeight[j + i * prevSize] * deltaPrev[j+ (idx*prevSize)]);
        }
        deltaCurrent[i + (idx*curSize)] = error * currentLayer[i+ (idx*curSize)] * (1 - currentLayer[i+ (idx*curSize)]);
    }

}

__global__ void kBackProp(float * curBias, float * deltaCur, float * curWeights, float * prevLayer, float learnRate, int curSize, int prevSize) {
        int idx = threadIdx.x;

        for(int i = 0; i < curSize; i++) {
        curBias[i] = curBias[i] - (learnRate * deltaCur[i + (idx*curSize)]);
        for(int j = 0; j < prevSize; j ++) {
            curWeights[j + i * prevSize] = curWeights[j + i * prevSize] - (learnRate * deltaCur[i + (idx*curSize)]) * prevLayer[j + (idx*prevSize)];
        }
    } 
}

__global__ void gpuTrainNetwork(float * inLayer, float * hLayerOne, float * outLayer, float * oneWeights, float * outWeights, float * oneBias,
                 float * outBias, float * input, float * correctInput, int amountOfData, int inSize, int hiddenSize, int outSize, int epochs, float learnRate) {

    int idx = (blockDim.x *blockIdx.x) + threadIdx.x;
    cudaDeviceSynchronize();

    kActivate<<<1, amountOfData>>>(inLayer, hLayerOne, oneWeights, oneBias, inSize, hiddenSize);
    cudaDeviceSynchronize();
    kActivate<<<1, amountOfData>>>(hLayerOne, outLayer, outWeights, outBias, hiddenSize, outSize);
    cudaDeviceSynchronize();

    float * deltaOut = (float *) malloc(sizeof(float) * outSize * amountOfData);
    float * deltaOne = (float *) malloc(sizeof(float) * hiddenSize * amountOfData);

    kOutError<<<1, amountOfData>>> (deltaOut, correctInput, outLayer, outSize);
    cudaDeviceSynchronize();
    kError<<<1, amountOfData>>> (deltaOne, deltaOut, hLayerOne, outWeights, hiddenSize, outSize);
    cudaDeviceSynchronize();

    for(int z = 0; z < amountOfData; z++) {
        for(int i = 0; i < outSize; i++) {
            outBias[i] = outBias[i] - (learnRate * deltaOut[i + (z*outSize)]);
            for(int j = 0; j < hiddenSize; j++) {
                outWeights[j + i * hiddenSize] = outWeights[j + i * hiddenSize] - (learnRate * deltaOut[i + (z*outSize)]) * hLayerOne[j + (z*hiddenSize)];
            }
        } 

        
        for(int i = 0; i < hiddenSize; i++) {
            oneBias[i] = oneBias[i] - (learnRate * deltaOne[i + (z*hiddenSize)]);
            for(int j = 0; j < inSize; j ++) {
                oneWeights[j + i * inSize] = oneWeights[j + i * inSize] - (learnRate * deltaOne[i + (z*hiddenSize)]) * inLayer[j + (z*inSize)];
            }
        } 
    }

    for(int i = 0; i < amountOfData; i++) {
        printf("Input: %f %f ", inLayer[i * 2], inLayer[i * 2 + 1]);
        printf("Correct input: %f", correctInput[i]);
        printf("Hello: %f \n", outLayer[i]);
    }
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
