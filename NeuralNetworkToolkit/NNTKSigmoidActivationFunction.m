//
//  NNTKSigmoidActivationFunction.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/29/23.
//

#import "NNTKSigmoidActivationFunction.h"

@implementation NNTKSigmoidActivationFunction

- (double)activate:(double)value {
    return 1.0 / (1.0 + exp(-value));
}

- (double)derivative:(double)value {
    double activation = [self activate:value];
    return activation * (1.0 - activation);
}

- (double)derivativeFromOutput:(double)output {
    return output * (1.0 - output);
}

@end
