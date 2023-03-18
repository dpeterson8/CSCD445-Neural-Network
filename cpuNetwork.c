#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "cpuNetwork.h"


// sigmoid: function used to take sigmoid of only input: x
float sigmoid(float x) {

    return 1 / (1 + exp(-x));

}

float dSigmoid(float x) {

    return sigmoid(x) * (1 - sigmoid(x));

}

/*
    iniateWeights: Function used to iniate random weights and biases between 0.0 - 1.0;
    params: 
        inArr -- Input array of weights or biases to be iniated
        size -- Size of input array
*/
void iniateWeigts(float * inArr, int size) {
    
    for(int i = 0; i < size; i++) {
        inArr[i] = (((float)rand()/(float)RAND_MAX))/10;
    }

}

/*
    fillInputLayer: This function will populate the input layer with a given array at a given postion. 
                        The input layer will take the same amount of data as there is input nodes.
    params:
        input -- Given input to fill the input layer
        inputLayer -- Input layer to be populated with data
        inputSize -- Size of input layer used to get set amount of data from input
        position -- Position of current input array
*/
void fillInputLayer(float * input, float * inputLayer, int inputSize, int position) {
    int sizeOfLayer = (sizeof(inputLayer)/sizeof(inputLayer[0]));

    if(sizeof(input)/sizeof(input[0]) != sizeOfLayer) {
        printf("Error size of input and input layer not even");
        return;
    }

    for(int i = 0; i < inputSize; i++) {
        inputLayer[i] = input[i + inputSize * position];
    }

}

/*
    activateLayers: This function will take the current active layer and apply the
        activation function which is:
            sigmoid( ( active layer * next layer weights ) + bias )
    params:
        inputLayer -- Current active layer
        nextLayer -- Next active layer which will be either hidden or output layers
        nextWeight -- Weight that will be applied to each input value before adding the bias
        nextBias -- Biases used by the next layer
*/
void activateLayers( float * inputLayer, float * nextLayer, float * nextWeights, float * nextBias, int inputSize, int nextSize) {

    for(int i = 0; i < nextSize; i++) {
        float result = nextBias[i];

        for(int j = 0; j < inputSize; j++) {
            result += inputLayer[j] * nextWeights[j + i * inputSize];
        }

        nextLayer[i] = sigmoid(result);
    }

}

/*
    outError: This function will calculate the given error based off of the output nodes and expected output.
    
    params:
        deltaOut -- The array used to store the calculated errors
        correctInput -- The array mathing the expected results of the output layer
        outLayer -- The output layer
        outSize -- Size of output layer
        dataPosition -- Position of expected output array
*/
void outError(float * deltaOut, float * correctInput, float * outLayer, int outSize, int dataPosition) {
    for(int i = 0; i < outSize; i++) {
        deltaOut[i] = outLayer[i] * (1 - outLayer[i]) * (outLayer[i] - correctInput[i + outSize * dataPosition]);
    }

}

void testOutError(float * deltaOut, float * correctInput, float * outLayer, int outSize, int dataPosition) {
    for(int i = 0; i < outSize; i++) {
        deltaOut[i] = (correctInput[i] - outLayer[i]) * dSigmoid(outLayer[i]);
    }

}

/*
    outError: This function will calculate the given error based off of the two hidden layers or input and hidden 
    
    params:
        deltaCurrem -- The array used to store the calculated errors
        deltaPrec -- The array holding errors from the last calculated layer
        currentLayer -- Current layer being traversed
        prevLayerWeight -- Weights of the previous layer ex: layer 2 active, then prevLayer = hiden layer 1
        curSize -- Size of current layer
        prevSize -- Size of previous layer
*/
void hiddenError(float * deltaCurrent, float * deltaPrev, float * currentLayer, float * prevLayerWeight, int curSize, int prevSize) {
    for(int i = 0; i < curSize; i++) {
        float error = 0.0;
        for(int j = 0; j < prevSize; j++) {
            error += (prevLayerWeight[j + i * prevSize] * deltaPrev[j]);
        }
        deltaCurrent[i] = error * currentLayer[i] * (1 - currentLayer[i]);
    }
}

void testHiddenError(float * deltaCurrent, float * deltaPrev, float * currentLayer, float * prevLayerWeight, int curSize, int prevSize) {
    for(int i = 0; i < curSize; i++) {
        float error = 0.0;
        for(int j = 0; j < prevSize; j++) {
            error += (prevLayerWeight[j + i * prevSize] * deltaPrev[j]);
        }
        deltaCurrent[i] = error * dSigmoid(currentLayer[i]);
    }
}


/*
    backProp: This function will work backwords on the network and apply the calculated errors to adjust the 
                current weights and biases.
    params:
        curBias -- Bias array at current layer
        deltaCur -- Error according to current layer
        curWeights -- Weights array at current layer
        prevLayer -- Previous layer array
        learnRate -- Rate at which the network will learn at or increase/decrease weights and biases
        curSize -- Size of current layer
        prevSize -- Size of previous layer
*/
void backProp(float * curBias, float * deltaCur, float * curWeights, float * prevLayer, float learnRate, int curSize, int prevSize) {
    for(int i = 0; i < curSize; i++) {
        curBias[i] = curBias[i] - (learnRate * deltaCur[i]);
        for(int j = 0; j < prevSize; j ++) {
            curWeights[j + i * prevSize] = curWeights[j + i * prevSize] - (learnRate * deltaCur[i]) * prevLayer[j];
        }
    } 
}

void testBackProp(float * curBias, float * deltaCur, float * curWeights, float * prevLayer, float learnRate, int curSize, int prevSize) {
    for(int i = 0; i < curSize; i++) {
        curBias[i] = curBias[i] - (learnRate * deltaCur[i]);
        for(int j = 0; j < prevSize; j ++) {
            curWeights[j + i * prevSize] += learnRate * deltaCur[i] * prevLayer[j];
        }
    }  
}