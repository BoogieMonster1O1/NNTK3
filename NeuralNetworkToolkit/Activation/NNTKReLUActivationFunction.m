//
//  NNTKReLUActivationFunction.m
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/29/23.
//

#import "NNTKReLUActivationFunction.h"

@implementation NNTKReLUActivationFunction

- (double)activate:(double)value {
    return fmax(0.0, value);
}

- (double)derivative:(double)value {
    if (value <= 0.0) {
        return 0.0;
    } else {
        return 1.0;
    }
}

- (double)derivativeFromOutput:(double)output {
    if (output == 0.0) {
        return 0.0;
    } else {
        return 1.0;
    }
}

@end
