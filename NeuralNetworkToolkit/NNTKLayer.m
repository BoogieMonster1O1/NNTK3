//
//  Layer.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/28/23.
//

#import "NNTKLayer.h"
#include <stdlib.h>
#import <Accelerate/Accelerate.h>

@implementation NNTKLayer

- (instancetype)initWithActivationFunction:(id<NNTKActivationFunction>)activationFunction inputDimension:(NSUInteger)inputDimension outputDimension:(NSUInteger)outputDimension {
    self = [super init];
    if (self) {
        _activationFunction = activationFunction;
        _inputDimension = inputDimension;
        _outputDimension = outputDimension;
        _weightsMatrix = calloc(outputDimension * inputDimension, sizeof(float));
        _biasesVector = calloc(outputDimension, sizeof(float));
        _cachedOutput = NULL;
        _cachedUnactivatedOutput = NULL;
    }
    return self;
}

- (void)deallocBuffers {
    free(_weightsMatrix);
    free(_biasesVector);
    free(_cachedOutput);
    free(_cachedUnactivatedOutput);
}

- (float *)forward:(float *)inputVector {
    float *outputVector = calloc(_outputDimension, sizeof(float));
    
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                (int)_outputDimension, (int)_inputDimension,
                1.0, _weightsMatrix, (int)_inputDimension,
                inputVector, 1,
                1.0, outputVector, 1);
    
    for (NSUInteger i = 0; i < _outputDimension; i++) {
        outputVector[i] += _biasesVector[i];
        outputVector[i] = [_activationFunction activate:outputVector[i]];
    }
    
    return outputVector;
}

@end
