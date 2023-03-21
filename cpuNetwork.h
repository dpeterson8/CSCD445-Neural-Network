#ifndef CPUNETWORK_H_
#define CPUNETWORK_H_

float sigmoid(float x);
float dSigmoid(float x);
void iniateWeigts(float * inArr, int size);
void fillInputLayer(float * input, float * inputLayer, int inputSize, int position);
void activateLayers( float * inputLayer, float * nextLayer, float * nextWeights, float * nextBias, int inputSize, int nextSize);
void outError(float * deltaOut, float * correctInput, float * outLayer, int outSize, int dataPosition);
void hiddenError(float * deltaCurrent, float * deltaPrev, float * currentLayer, float * prevLayerWeight, int curSize, int prevSize);
void testHiddenError(float * deltaCurrent, float * deltaPrev, float * currentLayer, float * prevLayerWeight, int curSize, int prevSize);
void backProp(float * curBias, float * deltaCur, float * curWeights, float * prevLayer, float learnRate, int curSize, int prevSize);
void testBackProp(float * curBias, float * deltaCur, float * curWeights, float * prevLayer, float learnRate, int curSize, int prevSize);
void trainNetwork(float * inLayer, float * hLayerOne, float * hLayerTwo, float * outLayer, float * oneWeights, float * twoWeights, float * outWeights, float * oneBias, float * twoBias,
                 float * outBias, float * input, float * correctInput, int amountOfData, int inSize, int hiddenSize, int outSize, int epochs, float learnRate, int total);


#endif