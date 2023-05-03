//
//  NSMutableData+NNTKFloatUtils.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 5/3/23.
//

#import "NSMutableData+NNTKFloatUtils.h"

@implementation NSMutableData (NNTKFloatUtils)

- (float *)mutableFloats {
    return (float *) [self mutableBytes];
}

- (NSUInteger)numberOfFloats {
    return [self length] / sizeof(float);
}

@end
