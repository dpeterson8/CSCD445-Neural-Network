#include <stdio.h>
#include <math.h>
#include <stdlib.h>

int INPUTSIZE = 2;
int HIDDENSIZE = 6;
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
    getMnist: This function will take the "mnist_train.scv" file and spilt the data within
                into two arrays the input for the network and the current inputs correct output.
    params: 
        inputArr -- Array to be populated with input data
        correctInput -- Array that carrys expected output of current input
*/
void getMnist(float * inputArr, float * correctInput) {
        FILE* ptr;
    int ch;
    int symb;
    int prevSymb = 44;
    int inputIndex = 0;
    int correctIndex = 0;
    int currentPixel = 0;

    ptr = fopen("mnist_train.csv", "r");

    while ( ( symb = getc( ptr ) ) != EOF && symb != 10) {  }

    while ( ( symb = getc( ptr ) ) != EOF && inputIndex < (784*60000)) {
        ch = symb - 48;
        // printf("%d ", symb);

        if(symb >= 48 && symb <= 57) {
            ch = symb - 48;

            if(prevSymb >= 48 && prevSymb <= 57) {
                inputIndex--;
                inputArr[inputIndex] = (inputArr[inputIndex] * 10) + ch;
                inputIndex++;
            } else {

                if(currentPixel % 785 == 0) {
                    // printf("%d ", symb-48);
                    correctInput[correctIndex] = ch;
                    correctIndex++; 
                    currentPixel++;
                } else {
                    inputArr[inputIndex] = ch;    
                    currentPixel++;
                    inputIndex++;
                } 

            // increase index counter
            // printf("%d ", symb - 48);
            }

            // inputIndex++;
        }

        prevSymb = symb;
    }
    fclose(ptr);
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
    for(int t = 0; t < outSize; t++) {
        deltaOut[t] = (correctInput[dataPosition * OUTSIZE + t] - outLayer[t]) * dSigmoid(outLayer[t]);
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
    for(int t = 0; t < curSize; t++) {
        float error = 0.0;
        for(int n = 0; n <prevSize; n++) {
            error+=deltaPrev[n] * prevLayerWeight[n + t * curSize];
        }
        deltaCurrent[t] = error * dSigmoid(currentLayer[t]);
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
    for(int t = 0; t < curSize; t++) {
        curBias[t] += deltaCur[t] * learnRate;
        for(int n = 0; n < prevSize; n++) {
            curWeights[n + t * prevSize] += prevLayer[n]*deltaCur[t]*learnRate;
        }               
    }  
}

int main() {
    srand(1129);

    float * inputArr = malloc(sizeof(float) * 784 * 60000);
    float * correctInput = malloc(sizeof(float) * 60000);

    getMnist(inputArr, correctInput);

    float * correctData = (malloc(sizeof(float) * 60000 * 10));

    for(int s = 0; s < 60000; s++) {
        for(int t = 0; t < 10; t++) {
            if(t == correctInput[s]) {
                correctData[s * (10) + t] = 1.0;
            } else {
                correctData[s * (10) + t] = 0.0;
            }
        }
    }
    
    float inputLayer[INPUTSIZE];
    float hiddenOneLayer[HIDDENSIZE];
    float hiddenTwoLayer[HIDDENSIZE];
    float outLayer[OUTSIZE];

    float hiddenLayerWeights[INPUTSIZE * HIDDENSIZE];
    float hiddenTwoLayerWeights[INPUTSIZE * HIDDENSIZE];
    float outLayerWights[HIDDENSIZE * OUTSIZE];

    float hiddenLayerBias[HIDDENSIZE];
    float hiddenTwoLayerBias[HIDDENSIZE];
    float outputLayerBias[OUTSIZE];

    float lr = 0.2;
    int epochs = 12000;
    static const int amountOfData = 4;

    // iniate hidden layer weights and biases
    iniateWeigts(hiddenLayerWeights, (INPUTSIZE * HIDDENSIZE));
    iniateWeigts(hiddenLayerBias, (HIDDENSIZE));

    // iniate out layer weights and biases
    iniateWeigts(outLayerWights, (HIDDENSIZE * OUTSIZE));
    iniateWeigts(outputLayerBias, (OUTSIZE));

    float test[4*2] = { 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    float inputCorrect[4] = {0.0,1.0,1.0,0.0}; 

    
    for(int i = 0; i < epochs; i++) {

        for(int j = 0; j < amountOfData; j++) {

            fillInputLayer(test, inputLayer, INPUTSIZE, j);
            // printArray(inputLayer, (INPUTSIZE));

            // Pass input layer values to hidden layer after activation function
            activateLayers(inputLayer, hiddenOneLayer, hiddenLayerWeights, hiddenLayerBias, INPUTSIZE, HIDDENSIZE);
            activateLayers(hiddenOneLayer, hiddenTwoLayer, hiddenTwoLayerWeights, hiddenTwoLayerBias, HIDDENSIZE, HIDDENSIZE);
            activateLayers(hiddenTwoLayer, outLayer, outLayerWights, outputLayerBias, HIDDENSIZE, OUTSIZE);


            float deltaOut[OUTSIZE];
            float deltaTwoHidden[HIDDENSIZE];
            float deltaOneHidden[HIDDENSIZE];

            outError(deltaOut,inputCorrect,outLayer,OUTSIZE, j);
            hiddenError(deltaTwoHidden, deltaOut, hiddenTwoLayer, outLayerWights, HIDDENSIZE, OUTSIZE);
            hiddenError(deltaOneHidden, deltaTwoHidden, hiddenOneLayer, hiddenTwoLayerWeights, HIDDENSIZE, HIDDENSIZE);

            backProp(outputLayerBias, deltaOut, outLayerWights, hiddenTwoLayer, lr, OUTSIZE, HIDDENSIZE);
            backProp(hiddenTwoLayer, deltaTwoHidden, hiddenTwoLayerWeights, hiddenOneLayer, lr, HIDDENSIZE, HIDDENSIZE);
            backProp(hiddenLayerBias, deltaOneHidden, hiddenLayerWeights, inputLayer, lr, HIDDENSIZE, INPUTSIZE);
            
            printArray(outLayer, (OUTSIZE));
            printf("Correct awnser: %f", inputCorrect[j]);
            printf("\n");
            // printArray(correctData + (j*10), OUTSIZE);
            // for(int z = 0; z < 784; z++) {
            //     if(z % 28 == 0) { printf("\n"); }
            //     printf("%d ", (int)inputLayer[z]);
            // }
            // printf("\n");
            // printf("\n");

        }
    }
    free(inputArr);
    free(correctInput);
}