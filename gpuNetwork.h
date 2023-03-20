#ifndef GPUNETWORK_H_
#define GPUNETWORK_H_

__device__ float gpuSigmoid(float x);

__global__ void gpuFillInputLayer(float * input, float * inputLayer, int inputSize, int position);

__global__ void gpuActivateLayers( float * layers, float * nextLayer, float * nextWeights, float * nextBias, int inputSize, int nextSize);


#endif