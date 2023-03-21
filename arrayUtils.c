#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "arrayUtils.h"


void shuffle(float * arr, float * arrCor, int arrSize, int inSize) {

    if (arrSize > 1) {
        int i;
        for(i = 0; i < arrSize; i++) {
            int j = i + rand() / (RAND_MAX / (arrSize-i) + 1);
            
                float temp1 = arr[j*inSize];
                float temp2 = arr[j*inSize+1];
                arr[j*inSize] = arr[i*inSize];
                arr[j*inSize+1] = arr[i*inSize+1];
                arr[i*inSize] = temp1;
                arr[i*inSize+1] = temp2;

                float corTemp = arrCor[j];
                arrCor[j] = arrCor[i];
                arrCor[i] = corTemp;

        }
    }
}

void printArray(float * array, int size) {

    for(int i = 0; i < size; i++) {
        printf(" %2f", array[i]);
    }

}


void totalCorrect(float * out, float * correct, int total, int outSize) {
    int greatestTrue = 11;
    int greatestGuess = 0;

    for(int z = 0; z < outSize; z++) {
        if(out[z] > out[greatestGuess]) {
            greatestGuess = z;
        }

        if(correct[z] > correct[greatestTrue]) {
            greatestTrue = z;
        }
    }

    if(greatestGuess == greatestTrue) {
        total++;
    }
}