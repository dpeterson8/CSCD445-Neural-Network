#ifndef GPUNETWORK_H_
#define GPUNETWORK_H_

__device__ float gpuSigmoid(float x);
__global__ void gpuFillInputLayer(float * input, float * inputLayer, int inputSize, int position);
__global__ void gpuActivateLayers( float * layers, float * nextLayer, float * nextWeights, float * nextBias, int inputSize, int nextSize);
__global__ void gpuOutError(float * deltaOut, float * correctInput, float * outLayer, int outSize, int dataPosition);
__global__ void gpuHiddenError(float * deltaCurrent, float * deltaPrev, float * currentLayer, float * prevLayerWeight, int curSize, int prevSize);
__global__ void gpuBackProp(float * curBias, float * deltaCur, float * curWeights, float * prevLayer, float learnRate, int curSize, int prevSize);
__global__ void gpuBackPropTwo(float * curBias, float * deltaCur, float * curWeights, float * prevLayer, float learnRate, int curSize, int prevSize);
__global__ void gpuTrainNetwork(float * inLayer, float * hLayerOne, float * outLayer, float * oneWeights, float * outWeights, float * oneBias, float * outBias, float * input, float * correctInput, int amountOfData, int inSize, int hiddenSize, int outSize, int epochs, float learnRate);

#endif