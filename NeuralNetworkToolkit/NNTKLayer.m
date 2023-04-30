//
//  Layer.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/28/23.
//

#import "NNTKLayer.h"
#include <stdlib.h>
#import <Accelerate/Accelerate.h>

static bool caching = false;

@implementation NNTKLayer

+ (void)setCaching:(bool)value {
    caching = value;
}

- (instancetype)initWithActivationFunction:(id<NNTKActivationFunction>)activationFunction inputDimension:(NSUInteger)inputDimension outputDimension:(NSUInteger)outputDimension {
    self = [super init];
    if (self) {
        _activationFunction = activationFunction;
        _inputDimension = inputDimension;
        _outputDimension = outputDimension;
        _weightsMatrix = calloc(outputDimension * inputDimension, sizeof(float));
        _biasesVector = calloc(outputDimension, sizeof(float));
        if (caching) {
            _cachedOutput = malloc(outputDimension * sizeof(float));
            _cachedUnactivatedOutput = malloc(outputDimension * sizeof(float));
            _cachedInput = malloc(inputDimension * sizeof(float));
        } else {
            _cachedOutput = NULL;
            _cachedUnactivatedOutput = NULL;
            _cachedInput = NULL;
        }
    }
    return self;
}

- (void)deallocBuffers {
    free(_weightsMatrix);
    free(_biasesVector);
    free(_cachedOutput);
    free(_cachedUnactivatedOutput);
    free(_cachedInput);
}

- (float *)forward:(float *)inputVector {
    if (caching) {
        return [self forwardCached:inputVector];
    }
    
    float *outputVector = calloc(_outputDimension, sizeof(float));
    
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int) _outputDimension, (int) _inputDimension, 1.0, _weightsMatrix, (int) _inputDimension, inputVector, 1, 1.0, outputVector, 1);
    
    for (NSUInteger i = 0; i < _outputDimension; i++) {
        outputVector[i] += _biasesVector[i];
        outputVector[i] = [_activationFunction activate:outputVector[i]];
    }
    
    return outputVector;
}

- (float *)forwardCached:(float *)inputVector {
    
    float *outputVector = calloc(_outputDimension, sizeof(float));
    
    cblas_sgemv(CblasRowMajor, CblasNoTrans, (int) _outputDimension, (int) _inputDimension, 1.0, _weightsMatrix, (int) _inputDimension, inputVector, 1, 1.0, outputVector, 1);
    
    for (NSUInteger i = 0; i < _outputDimension; i++) {
        outputVector[i] += _biasesVector[i];
        _cachedUnactivatedOutput[i] = outputVector[i];
        outputVector[i] = [_activationFunction activate:outputVector[i]];
    }
    
    memcpy(_cachedOutput, outputVector, _outputDimension);
    memcpy(_cachedInput, inputVector, _inputDimension);
    
    return outputVector;
}

@end
