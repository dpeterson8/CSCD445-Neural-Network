#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>
// #include "mnistFileUtils.h"
#include "cpuNetwork.h"
#include "arrayUtils.h"

#include "gpuNetwork.h"

int INPUTSIZE = 2;
int HIDDENSIZE = 4;
int OUTSIZE = 1;


void testNetwork(int amountOfData, float * inputLayer, float * firstHidden, float * secondHidden, float * outLayer, float * firstWeights, float * secondWeights,
            float * outWeights, float * firstBias, float * secondBias, float * outBias, float * input) {
    
    int testPos = rand() % amountOfData;
    if(testPos >= amountOfData) {
        testPos = 0;
    }

    fillInputLayer(input, inputLayer, INPUTSIZE, testPos);

    activateLayers(inputLayer, firstHidden, firstWeights, firstBias, INPUTSIZE, HIDDENSIZE);
    activateLayers(firstHidden, secondHidden, secondWeights, secondBias, HIDDENSIZE, HIDDENSIZE);
    activateLayers(secondHidden, outLayer, outWeights, outBias, HIDDENSIZE, OUTSIZE);
}

int main() {
    // __time_t t;
    srand((unsigned) time(NULL));
    
    float inputLayer[INPUTSIZE];
    float hiddenOneLayer[HIDDENSIZE];
    float hiddenTwoLayer[HIDDENSIZE];
    float outLayer[OUTSIZE];

    float layers[INPUTSIZE + (HIDDENSIZE * 2) + OUTSIZE];
    float weights[(INPUTSIZE * HIDDENSIZE * 2) + (OUTSIZE * HIDDENSIZE)];
    float biases[(HIDDENSIZE * 2) + OUTSIZE];

    float hiddenLayerWeights[INPUTSIZE * HIDDENSIZE];
    float hiddenTwoLayerWeights[INPUTSIZE * HIDDENSIZE];
    float outLayerWights[HIDDENSIZE * OUTSIZE];

    float hiddenLayerBias[HIDDENSIZE];
    float hiddenTwoLayerBias[HIDDENSIZE];
    float outputLayerBias[OUTSIZE];

    float test[16] = { 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    float inputCorrect[8] = {0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0}; 


    int totalCorrectPasses = 0;
    float lr = 0.4;
    int epochs = 200000;
    static const int amountOfData = 8;

    // iniateWeigts(weights, (INPUTSIZE * HIDDENSIZE * 2) + (OUTSIZE * HIDDENSIZE));
    // iniateWeigts(biases, (HIDDENSIZE * 2) + OUTSIZE);

    // iniate hidden layer weights and biases
    iniateWeigts(hiddenLayerWeights, (INPUTSIZE * HIDDENSIZE));
    iniateWeigts(hiddenLayerBias, (HIDDENSIZE));

    // iniate out layer weights and biases
    iniateWeigts(outLayerWights, (HIDDENSIZE * OUTSIZE));
    iniateWeigts(outputLayerBias, (OUTSIZE));

    float * d_input;
    float * d_correct;
    size_t d_input_size = sizeof(float) * amountOfData;

    float * d_input_layer;
    size_t d_inputLayer_size = sizeof(float) * INPUTSIZE;

    float * d_hLayerOne;
    float * d_hLayerTwo;
    float * d_outLayer;

    float * d_fWeights;
    float * d_outWeights;

    float * d_fBias;
    float * d_outBias;

    float * d_deltaOut;

    float * d_deltaOne;

    int hiddenLayerSize = sizeof(float) * HIDDENSIZE;
    int outLayerSize = sizeof(float) * OUTSIZE;

    int hiddenWeightSize = sizeof(float) * HIDDENSIZE * INPUTSIZE;

    cudaMalloc((void**)&d_input, d_input_size);
    cudaMalloc((void**)&d_correct, (sizeof(float) * OUTSIZE * amountOfData));
    cudaMalloc((void**)&d_input_layer, d_inputLayer_size * amountOfData);
    cudaMalloc((void**)&d_fWeights, hiddenWeightSize);
    cudaMalloc((void**)&d_fBias, (sizeof(float) * HIDDENSIZE));
    cudaMalloc((void**)&d_hLayerOne, hiddenLayerSize * amountOfData);
    cudaMalloc((void**)&d_hLayerTwo, hiddenLayerSize);
    cudaMalloc((void**)&d_outWeights, (sizeof(float) * HIDDENSIZE * OUTSIZE));
    cudaMalloc((void**)&d_outBias, (sizeof(float) * OUTSIZE));
    cudaMalloc((void**)&d_outLayer, outLayerSize * amountOfData);    
    cudaMalloc((void**)&d_deltaOut, (sizeof(float) * OUTSIZE) * amountOfData);
    cudaMalloc((void**)&d_deltaOne, (sizeof(float) * HIDDENSIZE) * amountOfData);

    cudaMemcpy(d_input_layer, test, d_inputLayer_size * amountOfData, cudaMemcpyHostToDevice);
    cudaMemcpy(d_correct, inputCorrect, (sizeof(float) * OUTSIZE * amountOfData), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fWeights, hiddenLayerWeights, hiddenLayerSize * INPUTSIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fBias, hiddenLayerBias, hiddenLayerSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_outWeights, outLayerWights, (sizeof(float) * HIDDENSIZE * OUTSIZE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_outBias, outputLayerBias, outLayerSize, cudaMemcpyHostToDevice);

    int test2;

    float deltaOut[OUTSIZE];
    float deltaTwoHidden[HIDDENSIZE];
    float deltaOneHidden[HIDDENSIZE];


    for(int i = 0; i < epochs; i++) {
        gpuTrainNetwork<<<1, 1>>>(d_input_layer, d_hLayerOne, d_outLayer, d_fWeights, d_outWeights, d_fBias, d_outBias, d_input, d_correct, amountOfData,INPUTSIZE, HIDDENSIZE, OUTSIZE, epochs, lr);    
        cudaDeviceSynchronize();
    }


    // trainNetwork(inputLayer, hiddenOneLayer, hiddenTwoLayer, outLayer, hiddenLayerWeights, hiddenTwoLayerWeights, outLayerWights, hiddenLayerBias,
    //              hiddenTwoLayerBias, outputLayerBias, test, inputCorrect, amountOfData, INPUTSIZE, HIDDENSIZE, OUTSIZE, epochs, lr);    


  

    // for(int i = 0; i < epochs; i++) {

    //     // for(int j = 0; j < amountOfData; j++) {
    //     //     int testPos = rand() % amountOfData;
 
    //     //     if(testPos >= amountOfData) {
    //     //         testPos = 0;
    //     //     }

    //     //     gpuFillInputLayer<<<1, INPUTSIZE>>>(d_input, d_input_layer, INPUTSIZE, testPos);

    //     //     gpuActivateLayers<<<HIDDENSIZE, INPUTSIZE>>>(d_input_layer, d_hLayerOne, d_fWeights, d_fBias, INPUTSIZE, HIDDENSIZE);
    //     //     gpuActivateLayers<<<OUTSIZE, HIDDENSIZE>>>(d_hLayerOne, d_outLayer, d_outWeights, d_outBias, INPUTSIZE, HIDDENSIZE);

    //     //     gpuOutError<<<1, OUTSIZE>>>(d_deltaOut, d_correct, d_outLayer, OUTSIZE, testPos);
    //     //     gpuHiddenError<<<HIDDENSIZE, OUTSIZE>>>(d_deltaOne, d_deltaOut, d_hLayerOne, d_outWeights, HIDDENSIZE, OUTSIZE);

    //     //     gpuBackProp<<<1, OUTSIZE>>>(d_outBias, d_deltaOut, d_outWeights, d_hLayerOne, lr, OUTSIZE, HIDDENSIZE);
    //     //     gpuBackProp<<<1, HIDDENSIZE>>>(d_fBias, d_deltaOne, d_fWeights, d_input_layer, lr, HIDDENSIZE, INPUTSIZE);
    //     //     // cudaDeviceSynchronize();

    //     //     // printf("%d, \n", inputLayer);

    //     // }
    //     // cudaMemcpy(hiddenOneLayer, d_hLayerOne, hiddenLayerSize, cudaMemcpyDeviceToHost);
    //     // cudaMemcpy(outLayer, d_outLayer, outLayerSize, cudaMemcpyDeviceToHost);
    //     // cudaMemcpy(deltaOut, d_deltaOut, outLayerSize, cudaMemcpyDeviceToHost);
    //     // cudaMemcpy(inputLayer, d_input_layer, d_inputLayer_size, cudaMemcpyDeviceToHost);
    //     // printArray(deltaOut, outLayerSize);
    //     // // printf(", Output: ");
    //     // // printArray(outLayer, OUTSIZE);
    //     // printf("\n");
    // }


        //     cudaMemcpy(outLayer, d_outLayer, outLayerSize, cudaMemcpyDeviceToHost);
        // cudaMemcpy(inputLayer, d_input_layer, d_inputLayer_size, cudaMemcpyDeviceToHost);
        // printArray(inputLayer, INPUTSIZE);
        // printf(", Output: ");
        // printArray(outLayer, OUTSIZE);
        // printf("\n");

}

// int main() {
//     __time_t t;
//     srand((unsigned) time(NULL));

//     float * inputArr = (float *) malloc(sizeof(float) * 784 * 60000);
//     float * correctInput = (float *) malloc(sizeof(float) * 60000);
//     float * correctData = (float *) (malloc(sizeof(float) * 60000 * 10));

//     getMnistTrain(inputArr, correctInput, correctData, 1);
    
//     float inputLayer[INPUTSIZE];
//     float hiddenOneLayer[HIDDENSIZE];
//     float hiddenTwoLayer[HIDDENSIZE];
//     float outLayer[OUTSIZE];

//     float layers[INPUTSIZE + (HIDDENSIZE * 2) + OUTSIZE];
//     float weights[(INPUTSIZE * HIDDENSIZE * 2) + (OUTSIZE * HIDDENSIZE)];
//     float biases[(HIDDENSIZE * 2) + OUTSIZE];

//     float hiddenLayerWeights[INPUTSIZE * HIDDENSIZE];
//     float hiddenTwoLayerWeights[INPUTSIZE * HIDDENSIZE];
//     float outLayerWights[HIDDENSIZE * OUTSIZE];

//     float hiddenLayerBias[HIDDENSIZE];
//     float hiddenTwoLayerBias[HIDDENSIZE];
//     float outputLayerBias[OUTSIZE];

//     float test[16] = { 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
//     float inputCorrect[8] = {0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0}; 


//     int totalCorrectPasses = 0;
//     float lr = 0.3;
//     int epochs = 30000;
//     static const int amountOfData = 8;

//     // iniateWeigts(weights, (INPUTSIZE * HIDDENSIZE * 2) + (OUTSIZE * HIDDENSIZE));
//     // iniateWeigts(biases, (HIDDENSIZE * 2) + OUTSIZE);

//     // iniate hidden layer weights and biases
//     iniateWeigts(hiddenLayerWeights, (INPUTSIZE * HIDDENSIZE));
//     iniateWeigts(hiddenLayerBias, (HIDDENSIZE));

//     // iniate out layer weights and biases
//     iniateWeigts(outLayerWights, (HIDDENSIZE * OUTSIZE));
//     iniateWeigts(outputLayerBias, (OUTSIZE));

//     float * d_input;
//     size_t d_input_size = sizeof(float) * INPUTSIZE * amountOfData;
//     cudaMemcpy(d_input, test, d_input_size, cudaMemcpyHostToDevice);
//     float * d_layer = malloc();
//     size_t d_layer_size = sizeof(float) * (INPUTSIZE + (HIDDENSIZE * 2) + OUTSIZE);
// //     float * d_weights;
//     float * d_biases;
//     cudaMalloc((void**)&d_input, d_input_size);
//     cudaMalloc((void**)&d_layer, d_layer_size);


//     int test2;
    // cudaMemcpy(d_layer, layers, d_layer_size, cudaMemcpyHostToDevice);

    
//     for(int i = 0; i < epochs; i++) {
//         totalCorrectPasses = 0;
//         // int testPos = rand() % amountOfData;
//         // randomizeArray(inputArr, correctData, 60000);
//         for(int j = 0; j < amountOfData; j++) {
//             int testPos = rand() % amountOfData;
//             // int testPos = j;
//             if(testPos >= amountOfData) {
//                 testPos = 0;
//             }

//             // printf("Hello, ");
//             // gpuFillInputLayer<<<1, INPUTSIZE>>>(d_input, d_layer, INPUTSIZE, testPos);
//             // cudaDeviceSynchronize();
//             fillInputLayer(test, inputLayer, INPUTSIZE, testPos);
//             // cudaMemcpy(layers, d_layer, d_layer_size, cudaMemcpyDeviceToHost);
//             // printArray(layers, (INPUTSIZE));

//             activateLayers(inputLayer, hiddenOneLayer, hiddenLayerWeights, hiddenLayerBias, INPUTSIZE, HIDDENSIZE);
//             activateLayers(hiddenOneLayer, hiddenTwoLayer, hiddenTwoLayerWeights, hiddenTwoLayerBias, HIDDENSIZE, HIDDENSIZE);
//             activateLayers(hiddenTwoLayer, outLayer, outLayerWights, outputLayerBias, HIDDENSIZE, OUTSIZE);

//             // testNetwork(amountOfData, inputLayer, hiddenOneLayer, hiddenTwoLayer, outLayer, hiddenLayerWeights, hiddenTwoLayerWeights, outLayerWights,
//             //                     hiddenLayerBias, hiddenTwoLayerBias, outputLayerBias, test );

//             float deltaOut[OUTSIZE];
//             float deltaTwoHidden[HIDDENSIZE];
//             float deltaOneHidden[HIDDENSIZE];


//             outError(deltaOut, inputCorrect, outLayer, OUTSIZE, testPos);
//             hiddenError(deltaTwoHidden, deltaOut, hiddenTwoLayer, outLayerWights, HIDDENSIZE, OUTSIZE);
//             hiddenError(deltaOneHidden, deltaTwoHidden, hiddenOneLayer, hiddenTwoLayerWeights, HIDDENSIZE, HIDDENSIZE);

//             backProp(outputLayerBias, deltaOut, outLayerWights, hiddenTwoLayer, lr, OUTSIZE, HIDDENSIZE);
//             backProp(hiddenTwoLayer, deltaTwoHidden, hiddenTwoLayerWeights, hiddenOneLayer, lr, HIDDENSIZE, HIDDENSIZE);
//             backProp(hiddenLayerBias, deltaOneHidden, hiddenLayerWeights, inputLayer, lr, HIDDENSIZE, INPUTSIZE);
            
//             test2 = testPos;

//             // totalCorrect(outLayer, inputCorrect, totalCorrectPasses, OUTSIZE);
//         }

//         // printf("\n");

//         printArray(inputLayer, (INPUTSIZE));
//         printArray(outLayer, (OUTSIZE));
//         printf(" Total: %d", totalCorrectPasses);
//         // double acc = (totalCorrect/(epochs*amountOfData));
//         // printf(" Accuracy: %f", acc);
//         // printf("\n");
//         // printArray(correctData + (test2*10), OUTSIZE);
//         // for(int z = 0; z < INPUTSIZE; z++) {
//         //     if(z % 28 == 0) { printf("\n"); }
//         //     printf("%d ", (int)inputLayer[z]);
//         // }
//         printf("\n");

//     }

//     // trainNetwork(layers, weights, biases, test, inputCorrect, amountOfData, INPUTSIZE, HIDDENSIZE, OUTSIZE, epochs, lr);
//     free(inputArr);
//     free(correctInput);

// }






