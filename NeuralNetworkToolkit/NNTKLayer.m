//
//  Layer.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/28/23.
//

#import "NNTKLayer.h"
#include <stdlib.h>
#import <Accelerate/Accelerate.h>
#import <NeuralNetworkToolkit/NSMutableData+NNTKFloatUtils.h>

static bool caching = false;

@implementation NNTKLayer

+ (void)setCaching:(bool)value {
    caching = value;
}

- (instancetype)initWithActivationFunction:(id<NNTKActivationFunction>)activationFunction inputDimension:(NSUInteger)inputDimension outputDimension:(NSUInteger)outputDimension {
    NSMutableData *weights = [[NSMutableData alloc] initWithCapacity:(outputDimension * inputDimension * sizeof(float))];
    NSMutableData *biases = [[NSMutableData alloc] initWithCapacity:(outputDimension * sizeof(float))];
    return [self initWithWeights:weights biases:biases activationFunction:activationFunction inputDimension:inputDimension outputDimension:outputDimension];
}

- (instancetype)initWithWeights:(NSMutableData *)weights biases:(NSMutableData *)biases activationFunction:(id<NNTKActivationFunction>)activationFunction inputDimension:(NSUInteger)inputDimension outputDimension:(NSUInteger)outputDimension {
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
    _cachedOutput = nil;
    _cachedUnactivatedOutput = nil;
    _cachedInput = nil;
}

- (NSData *)forward:(NSData *)inputVector {
    if (caching) {
        return [self forwardCached:inputVector];
    }
    
    NSMutableData *outputVector = [[NSMutableData alloc] initWithCapacity:(_outputDimension * sizeof(float))];
    float *rawOutput = [outputVector mutableFloats];
    
    cblas_sgemv(
                CblasRowMajor,
                CblasNoTrans,
                (int) _outputDimension,
                (int) _inputDimension, 1.0,
                [_weightsMatrix mutableFloats],
                (int) _inputDimension,
                (const float *)[inputVector bytes],
                1,
                1.0,
                rawOutput,
                1
                );
    
    float *rawBiases = [_biasesVector mutableFloats];
    
    for (NSUInteger i = 0; i < _outputDimension; i++) {
        rawOutput[i] += rawBiases[i];
        rawOutput[i] = [_activationFunction activate:rawOutput[i]];
    }
    
    return outputVector;
}

- (NSData *)forwardCached:(NSData *)inputVector {
    NSMutableData *outputVector = [[NSMutableData alloc] initWithCapacity:(_outputDimension * sizeof(float))];
    float *rawOutput = [outputVector mutableFloats];
    
    cblas_sgemv(
                CblasRowMajor,
                CblasNoTrans,
                (int) _outputDimension,
                (int) _inputDimension, 1.0,
                [_weightsMatrix mutableFloats],
                (int) _inputDimension,
                (const float *)[inputVector bytes],
                1,
                1.0,
                [outputVector mutableFloats],
                1
                );
    
    float *rawBiases = [_biasesVector mutableFloats];
    
    for (NSUInteger i = 0; i < _outputDimension; i++) {
        rawOutput[i] += rawBiases[i];
    }
    
    _cachedUnactivatedOutput = [[NSData alloc] initWithData:outputVector];
    
    for (NSUInteger i = 0; i < _outputDimension; i++) {
        rawOutput[i] = [_activationFunction activate:rawOutput[i]];
    }
    
    
    return outputVector;
}

@end
