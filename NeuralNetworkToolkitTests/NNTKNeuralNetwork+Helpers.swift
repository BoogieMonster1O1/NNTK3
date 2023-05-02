//
//  NNTKNeuralNetwork+Helpers.swift
//  NeuralNetworkToolkitTests
//
//  Created by Shrish Deshpande on 5/2/23.
//

import Foundation
import NeuralNetworkToolkit

extension NNTKNeuralNetwork {
    func getLayer(at index: Int) -> NNTKLayer {
        return self.layers[index] as! NNTKLayer
    }
    
    func dumpLayerThings() {
        for layer in self.layers {
            print("\((layer as! NNTKLayer).inputDimension) -> \((layer as! NNTKLayer).outputDimension)")
        }
    }
}
