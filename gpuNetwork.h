#ifndef GPUNETWORK_H_
#define GPUNETWORK_H_

__device__ float gpuSigmoid(float x);

__global__ void gpuFillInputLayer(float * input, float * inputLayer, int inputSize, int position);


#endif