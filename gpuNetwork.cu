#include <math.h>

#include "gpuNetwork.h"

__device__ float gpuSigmoid(float x) {
    return 1 / (1 + exp(-x));
}

__global__ void gpuFillInputLayer(float * input, float * inputLayer, int inputSize, int position) {
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = (iy * (gridDim.x * blockDim.x)) + ix;

    inputLayer[idx] = input[idx + inputSize * position];
    
}

// void gpuActivateLayers( float * inputLayer, float * nextLayer, float * nextWeights, float * nextBias, int inputSize, int nextSize) {

//     for(int i = 0; i < nextSize; i++) {
//         float result = nextBias[i];

//         for(int j = 0; j < inputSize; j++) {
//             result += inputLayer[j] * nextWeights[j + i * inputSize];
//         }

//         nextLayer[i] = sigmoid(result);
//     }

// }
