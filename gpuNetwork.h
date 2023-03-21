#ifndef GPUNETWORK_H_
#define GPUNETWORK_H_

__global__ void gpuTrainNetwork(float * inLayer, float * hLayerOne, float * outLayer, float * oneWeights, float * outWeights, float * oneBias, float * outBias, float * input, float * correctInput, int amountOfData, int inSize, int hiddenSize, int outSize, int epochs, float learnRate, float * totalCorrect);

#endif