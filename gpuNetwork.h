#ifndef GPUNETWORK_H_
#define GPUNETWORK_H_

__device__ float gpuSigmoid(float x);

__global__ void gpuFillInputLayer(float * input, float * inputLayer, int inputSize, int position);

__global__ void gpuActivateLayers( float * layers, float * nextLayer, float * nextWeights, float * nextBias, int inputSize, int nextSize);

void trainNetwork(float * inLayer, float * hLayerOne, float * hLayerTwo, float * outLayer, float * oneWeights, float * twoWeights, float * outWeights, float * oneBias, float * twoBias,
                 float * outBias, float * input, float * correctInput, int amountOfData, int inSize, int hiddenSize, int outSize, int epochs, float learnRate);


#endif