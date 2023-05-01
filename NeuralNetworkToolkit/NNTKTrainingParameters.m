//
//  NNTKTrainingParameters.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 5/1/23.
//

#import "NNTKTrainingParameters.h"

@implementation NNTKTrainingParameters

- (instancetype)initWithEpochCount:(NSUInteger)epochCount learningRate:(float)learningRate {
    self = [super init];
    
    if (self) {
        _epochCount = epochCount;
        _learningRate = learningRate;
    }
    
    return self;
}

@end
