//
//  Layer.h
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/28/23.
//

#import <Foundation/Foundation.h>
#import <NeuralNetworkToolkit/NNTKActivationFunction.h>

NS_ASSUME_NONNULL_BEGIN

@interface NNTKLayer : NSObject

@property (nonatomic, strong) id<NNTKActivationFunction> activationFunction;
@property (nonatomic, assign) NSUInteger inputDimension;
@property (nonatomic, assign) NSUInteger outputDimension;
@property (nonatomic, strong) NSMutableData *weightsMatrix;
@property (nonatomic, strong) NSMutableData *biasesVector;
@property (nonatomic, strong, nullable) NSData *cachedOutput;
@property (nonatomic, strong, nullable) NSData *cachedUnactivatedOutput;
@property (nonatomic, strong, nullable) NSData *cachedInput;

+ (void)setCaching:(bool)value;

- (instancetype)initWithActivationFunction:(id<NNTKActivationFunction>)activationFunction inputDimension:(NSUInteger)inputDimension outputDimension:(NSUInteger)outputDimension;

- (instancetype)initWithWeights:(NSMutableData *)weights biases:(NSMutableData *)biases activationFunction:(id<NNTKActivationFunction>)activationFunction inputDimension:(NSUInteger)inputDimension outputDimension:(NSUInteger)outputDimension;

- (NSData *)forward:(NSData *)inputVector;

@end

NS_ASSUME_NONNULL_END
