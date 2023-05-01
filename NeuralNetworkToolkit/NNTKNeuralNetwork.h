//
//  NNTKNeuralNetwork.h
//  NeuralNetworkToolkit
//
//  Created by Shrish Deshpande on 5/1/23.
//

#import <Foundation/Foundation.h>
#import "NNTKLayer.h"

NS_ASSUME_NONNULL_BEGIN

@interface NNTKNeuralNetwork : NSObject

@property (nonatomic, strong) NSArray<NNTKLayer *> *layers;

- (void)deallocBuffers;

@end

NS_ASSUME_NONNULL_END
