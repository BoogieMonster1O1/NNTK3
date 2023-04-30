//
//  Layer.h
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/28/23.
//

#import <Foundation/Foundation.h>
#import "Activation/NNTKActivationFunction.h"

NS_ASSUME_NONNULL_BEGIN

@interface NNTKLayer : NSObject

@property (nonatomic, strong) id<NNTKActivationFunction> activationFunction;
@property (nonatomic, assign) NSUInteger inputDimension;
@property (nonatomic, assign) NSUInteger outputDimension;
@property (nonatomic, assign) float *weightsMatrix;
@property (nonatomic, assign) float *biasesVector;
@property (nonatomic, assign) float *cachedOutput;
@property (nonatomic, assign) float *cachedUnactivatedOutput;
@property (nonatomic, assign) float *cachedInput;

+ (void)setCaching:(bool)value;

- (instancetype)initWithActivationFunction:(id<NNTKActivationFunction>)activationFunction inputDimension:(NSUInteger)inputDimension outputDimension:(NSUInteger)outputDimension;

- (float *)forward:(float *)inputVector;

- (void)deallocBuffers;

@end

NS_ASSUME_NONNULL_END
