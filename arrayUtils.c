#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "arrayUtils.h"

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