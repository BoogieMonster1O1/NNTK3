//
//  NNTKTrainingParameters.h
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 5/1/23.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface NNTKTrainingParameters : NSObject

@property (nonatomic, assign) NSUInteger epochCount;
@property (nonatomic, assign) float learningRate;

- (instancetype)initWithEpochCount:(NSUInteger)epochCount learningRate:(float)learningRate;

@end

NS_ASSUME_NONNULL_END
