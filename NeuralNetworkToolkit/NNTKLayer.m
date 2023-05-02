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
    float *weights = calloc(outputDimension * inputDimension, sizeof(float));
    float *biases = calloc(outputDimension, sizeof(float));
    return [self initWithWeights:weights biases:biases activationFunction:activationFunction inputDimension:inputDimension outputDimension:outputDimension];
}

- (instancetype)initWithWeights:(float *)weights biases:(float *)biases activationFunction:(id<NNTKActivationFunction>)activationFunction inputDimension:(NSUInteger)inputDimension outputDimension:(NSUInteger)outputDimension {
    self = [super init];
    if (self) {
        _activationFunction = activationFunction;
        _inputDimension = inputDimension;
        _outputDimension = outputDimension;
        _weightsMatrix = weights;
        _biasesVector = biases;
        [self initCaching];
    }
    return self;
}

- (void)initCaching {
    if (caching) {
        _cachedOutput = malloc(_outputDimension * sizeof(float));
        _cachedUnactivatedOutput = malloc(_outputDimension * sizeof(float));
        _cachedInput = malloc(_inputDimension * sizeof(float));
    } else {
        _cachedOutput = nil;
        _cachedUnactivatedOutput = nil;
        _cachedInput = nil;
    }
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
