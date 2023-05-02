//
//  NNTKNeuralNetwork.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 5/1/23.
//

#import "NNTKNeuralNetwork.h"
#import <NeuralNetworkToolkit/NNTKReLUActivationFunction.h>

@implementation NNTKNeuralNetwork

- (instancetype)initWithInputDimension:(NSUInteger)inputDimension outputDimension:(NSUInteger)outputDimension outputActivationFunction:(id<NNTKActivationFunction>)activationFunction {
    NNTKLayer *firstLayer = [[NNTKLayer alloc] initWithActivationFunction:activationFunction inputDimension:inputDimension outputDimension:outputDimension];
    return [self initWithLayers: [@[firstLayer] mutableCopy]];
}

- (instancetype)initWithInputDimension:(NSUInteger)inputDimension outputDimension:(NSUInteger)outputDimension {
    return [self initWithInputDimension:inputDimension outputDimension:outputDimension outputActivationFunction:[NNTKReLUActivationFunction new]];
}

- (instancetype)initWithLayers:(NSArray<NNTKLayer *> *)layers {
    self = [super init];
    
    if (self) {
        if ([layers count] < 1) {
            return nil;
        }
        
        _inputDimension = [[layers firstObject] inputDimension];
        _outputDimension = [[layers lastObject] outputDimension];
        _layerCount = [layers count];
        _layers = [layers mutableCopy];
    }
    
    return self;
}

- (void)addHiddenLayerWithOutputDimension:(NSUInteger)outputDimension activationFunction:(id<NNTKActivationFunction>)activationFunction {
    _layerCount += 1;
    NNTKLayer *lastLayer = [_layers lastObject];
    [_layers removeLastObject];
    NSUInteger hiddenLayerInput = _inputDimension;
    if ([_layers lastObject]) {
        hiddenLayerInput = [[_layers lastObject] outputDimension];
    }
    NNTKLayer *hiddenLayer = [[NNTKLayer alloc] initWithActivationFunction:activationFunction inputDimension:hiddenLayerInput outputDimension:outputDimension];
    NNTKLayer *newLastLayer = [[NNTKLayer alloc] initWithActivationFunction:[lastLayer activationFunction] inputDimension:outputDimension outputDimension:_outputDimension];
    [_layers addObject:hiddenLayer];
    [_layers addObject:newLastLayer];
    [lastLayer deallocBuffers];
}

- (void)addHiddenLayerWithOutputDimension:(NSUInteger)outputDimension {
    [self addHiddenLayerWithOutputDimension:outputDimension activationFunction:[NNTKReLUActivationFunction new]];
}

- (void)deallocBuffers {
    for (NSUInteger i = 0; i < _layerCount; i++) {
        [_layers[(int) i] deallocBuffers];
    }
}

@end
