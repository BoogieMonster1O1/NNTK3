//
//  Layer.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/28/23.
//

#import "NNTKLayer.h"
#include <stdlib.h>

@implementation NNTKLayer

- (instancetype)initWithActivationFunction:(id<NNTKActivationFunction>)activationFunction inputDimension:(NSUInteger)inputDimension outputDimension:(NSUInteger)outputDimension {
    self = [super init];
    if (self) {
        _activationFunction = activationFunction;
        _inputDimension = inputDimension;
        _outputDimension = outputDimension;
        _weightsMatrix = calloc(outputDimension * inputDimension, sizeof(float));
        _biasesVector = calloc(outputDimension, sizeof(float));
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
    return nullptr
}

@end
