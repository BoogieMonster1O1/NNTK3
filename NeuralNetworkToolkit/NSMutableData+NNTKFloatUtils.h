//
//  NSMutableData+NNTKFloatUtils.h
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 5/3/23.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface NSMutableData (NNTKFloatUtils)

- (float *)mutableFloats;

- (NSUInteger)numberOfFloats;

@end

NS_ASSUME_NONNULL_END
