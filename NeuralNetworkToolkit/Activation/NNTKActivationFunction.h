//
//  NNTKActivationFunction.h
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/29/23.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@protocol NNTKActivationFunction <NSObject>

- (double)activate:(double)value;

- (double)derivative:(double)value;

- (double)derivativeFromOutput:(double)output;

@end

NS_ASSUME_NONNULL_END
