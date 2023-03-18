#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mnistFileUtils.h"

void swap(float *a, float *b) {
    float temp = *a;

    *a = *b;
    *b = temp;

}

void randomizeArray(float * arr, float * correctArr, int arrSize) {
    srand(time(NULL));

    for(int i = arrSize-1; i > 0; i--) {
        int randIndex = rand() % (i+1);
        int randCorIndex = randIndex * 10;
        randIndex = randIndex * 784;
        int index = i * 784;
        int corI = i * 10;

        for(int j = 0; j < 784; j++) {
            swap(&arr[index + j], &arr[randIndex + j]);    
        } 

        for(int j = 0; j < 10; j++) {
            swap(&correctArr[corI + j], &correctArr[randCorIndex + j]);
        }
    }
}

/*
    getMnist: This function will take the "mnist_train.scv" file and spilt the data within
                into two arrays the input for the network and the current inputs correct output.
    params: 
        inputArr -- Array to be populated with input data
        correctInput -- Array that carrys expected output of current input
*/
void getMnistTrain(float * inputArr, float * correctInput, float * correctData, int option) {
    FILE* ptr;
    int ch;
    int symb;
    int arrSize; 
    int prevSymb = 44;
    int inputIndex = 0;
    int correctIndex = 0;
    int currentPixel = 0;

    if(option == 1) {
        ptr = fopen("mnist_train.csv", "r");
        arrSize = 60000;
    } else {
        ptr = fopen("mnist_test.csv", "r");
        arrSize = 10000;
    }

    while ( ( symb = getc( ptr ) ) != EOF && symb != 10) {}

    while ( ( symb = getc( ptr ) ) != EOF && inputIndex < (784*arrSize)) {
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
            }
        }
        prevSymb = symb;
    }
    fclose(ptr);

    if(option == 1) {
        for(int s = 0; s < 60000; s++) {
        for(int t = 0; t < 10; t++) {
            if(t == correctInput[s]) {
                correctData[s * (10) + t] = 1.0;
            } else {
                correctData[s * (10) + t] = 0.0;
            }
        }
    }
    }
}