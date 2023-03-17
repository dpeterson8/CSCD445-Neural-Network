#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>


int INPUTSIZE = 2;
int HIDDENSIZE = 4;
int OUTSIZE = 1;

// sigmoid: function used to take sigmoid of only input: x
float sigmoid(float x) {

    return 1 / (1 + exp(-x));

}

float dSigmoid(float x) {

    return x * (1 - x);

}

/*
    iniateWeights: Function used to iniate random weights and biases between 0.0 - 1.0;
    params: 
        inArr -- Input array of weights or biases to be iniated
        size -- Size of input array
*/
void iniateWeigts(float * inArr, int size) {
    
    for(int i = 0; i < size; i++) {
        inArr[i] = ((float)rand()/(float)RAND_MAX);
    }

}

void printArray(float * array, int size){

    for(int i = 0; i < size; i++) {
        printf(" %2f", array[i]);
    }

}

// float calcError(float * deltaOut, float * deltaHidden, float * outWeights, float * hiddenLayer, float * outLayer, float ** correctInput) {
    
//     for(int t = 0; t < OUTSIZE; t++) {
//         float error = correctInput[j][t] - outLayer[t];
//         deltaOut[t] = error * dSigmoid(outLayer[t]);
//     }

//     for(int t = 0; t < HIDDENSIZE; t++) {
//         float error = 0.0;
//         for(int n = 0; n <OUTSIZE; n++) {
//             error+=deltaOut[n] * outWeights[n + t * HIDDENSIZE];
//         }
//         deltaHidden[t] = error * dSigmoid(hiddenLayer[t]);
//     }

// }

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

void fillInputLayer(float * input, float * inputLayer, int inputSize, int position) {
    int sizeOfLayer = (sizeof(inputLayer)/sizeof(inputLayer[0]));

    if(sizeof(input)/sizeof(input[0]) != sizeOfLayer) {
        printf("Error size of input and input layer not even");
        return;
    }

    for(int i = 0; i < sizeOfLayer; i++) {
        inputLayer[i] = input[i + inputSize * position];
    }

}

int main() {
    
    float inputLayer[INPUTSIZE];
    float hiddenOneLayer[HIDDENSIZE];
    float outLayer[OUTSIZE];

    float hiddenLayerWeights[INPUTSIZE * HIDDENSIZE];
    float outLayerWights[HIDDENSIZE * OUTSIZE];

    float hiddenLayerBias[HIDDENSIZE];
    float outputLayerBias[OUTSIZE];

    float lr = 0.1;
    int epochs = 10000;
    static const int amountOfData = 4;
    // float test[4][2] = { {0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    // float inputCorrect[4][1] = {{0.0},{1.0},{1.0},{0.0}}; 

    float test[4*2] = { 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    float inputCorrect[4] = {0.0,1.0,1.0,0.0}; 


    // fillInputLayer(test[1], inputLayer);
    // printArray(inputLayer, (INPUTSIZE));

    // iniate hidden layer weights and biases
    iniateWeigts(hiddenLayerWeights, (INPUTSIZE * HIDDENSIZE));
    iniateWeigts(hiddenLayerBias, (HIDDENSIZE));

    // iniate out layer weights and biases
    iniateWeigts(outLayerWights, (HIDDENSIZE * OUTSIZE));
    iniateWeigts(outputLayerBias, (OUTSIZE));

    // printArray(hiddenLayerWeights, (INPUTSIZE * HIDDENSIZE));
    
    for(int i = 0; i < epochs; i++) {

        for(int j = 0; j < amountOfData; j++) {

            fillInputLayer(test, inputLayer, INPUTSIZE, j);
            printArray(inputLayer, (INPUTSIZE));

            // Pass input layer values to hidden layer after activation function
            activateLayers(inputLayer, hiddenOneLayer, hiddenLayerWeights, hiddenLayerBias, INPUTSIZE, HIDDENSIZE);
            // printf("\nHidden Layer: ");
            // printArray(hiddenOneLayer, (HIDDENSIZE));
            
            // Pass hidden layer values to out layer after activation function
            activateLayers(hiddenOneLayer, outLayer, outLayerWights, outputLayerBias, HIDDENSIZE, OUTSIZE);
            // printf("\n");
            // printArray(outLayer, (OUTSIZE));
            // printf("\n");

            float deltaOut[OUTSIZE];
            for(int t = 0; t < OUTSIZE; t++) {
                // float error = inputCorrect[j * OUTSIZE + t] - outLayer[t];
                deltaOut[t] = (inputCorrect[j * OUTSIZE + t] - outLayer[t]) * dSigmoid(outLayer[t]);
            }

            float deltaHidden[HIDDENSIZE];
            for(int t = 0; t < HIDDENSIZE; t++) {
                float error = 0.0;
                for(int n = 0; n <OUTSIZE; n++) {
                    error+=deltaOut[n] * outLayerWights[n + t * HIDDENSIZE];
                }
                deltaHidden[t] = error * dSigmoid(hiddenOneLayer[t]);
            }

            // calcError(deltaOut, deltaHidden, outLayerWights, hiddenOneLayer, outLayer, inputCorrect);

            for(int t = 0; t < OUTSIZE; t++) {
                outputLayerBias[t] += deltaOut[t] * lr;
                for(int n = 0; n < HIDDENSIZE; n++) {
                    outLayerWights[n + t * HIDDENSIZE] += hiddenOneLayer[n]*deltaOut[t]*lr;
                }               
            }


            for(int t = 0; t < HIDDENSIZE; t++) {
                hiddenLayerBias[t] += deltaHidden[t] * lr;
                for(int n = 0; n < INPUTSIZE; n++) {
                    hiddenLayerWeights[n + t * INPUTSIZE] += inputLayer[n]*deltaHidden[t]*lr;
                }               
            }

            printArray(outLayer, (OUTSIZE));
            printf("\n");

        }
    }

    // float * deltaOut[OUTSIZE];
    // float * delatHidden[HIDDENSIZE];
    
}