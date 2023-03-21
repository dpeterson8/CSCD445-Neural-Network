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

    cudaDeviceSynchronize();
    kActivate<<<1, amountOfData>>> (inLayer, hLayerOne, oneWeights, oneBias, inSize, hiddenSize);
    cudaDeviceSynchronize();
    kActivate<<<1, amountOfData>>> (hLayerOne, outLayer, outWeights, outBias, hiddenSize, outSize);
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

    for(int i = 0; i < outSize; i++) {
            printf ("Input: %f, %f ", inLayer[i * 2], inLayer[i * 2 + 1]);
            printf("Output: %f ", outLayer[i]);
    }
    printf("\n");

    
}
