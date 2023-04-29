//
//  Layer.h
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 4/28/23.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface NNTKLayer : NSObject

@property (nonatomic, assign) NSUInteger inputDimension;
@property (nonatomic, assign) NSUInteger outputDimension;
@property (nonatomic, assign) float *weightsMatrix;
@property (nonatomic, assign) float *biasesVector;

@end

NS_ASSUME_NONNULL_END
