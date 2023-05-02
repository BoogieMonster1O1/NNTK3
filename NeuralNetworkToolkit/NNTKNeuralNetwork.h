//
//  NNTKNeuralNetwork.h
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 5/1/23.
//

#import <Foundation/Foundation.h>
#import <NeuralNetworkToolkit/NNTKLayer.h>

NS_ASSUME_NONNULL_BEGIN

@interface NNTKNeuralNetwork : NSObject

@property (nonatomic, strong) NSMutableArray<NNTKLayer *> *layers;
@property (nonatomic, assign) NSUInteger inputDimension;
@property (nonatomic, assign) NSUInteger outputDimension;
@property (nonatomic, assign) NSUInteger layerCount;

- (instancetype)initWithInputDimension:(NSUInteger)inputDimension outputDimension:(NSUInteger)outputDimension outputActivationFunction:(id<NNTKActivationFunction>)activationFunction;

- (instancetype)initWithInputDimension:(NSUInteger)inputDimension outputDimension:(NSUInteger)outputDimension;

- (void)addHiddenLayerWithOutputDimension:(NSUInteger)outputDimension activationFunction:(id<NNTKActivationFunction>)activationFunction;

- (void)addHiddenLayerWithOutputDimension:(NSUInteger)outputDimension;

- (void)deallocBuffers;

@end

NS_ASSUME_NONNULL_END
