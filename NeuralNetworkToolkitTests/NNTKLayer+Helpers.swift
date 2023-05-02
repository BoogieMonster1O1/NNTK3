//
//  NNTKLayer+Helpers.swift
//  NeuralNetworkToolkitTests
//
//  Created by Shrish Deshpande on 5/2/23.
//

import Foundation
import NeuralNetworkToolkit
import XCTest

extension NNTKLayer {
    func assertInOut(_ inputDimension: UInt, _ outputDimension: UInt) {
        XCTAssertEqual(self.inputDimension, inputDimension, "Wrong input dimension")
        XCTAssertEqual(self.outputDimension, outputDimension, "Wrong output dimension")
    }
}

extension NNTKLayer {
    func printWeightsMatrix() {
        print("Weights Matrix")
        for i in 0..<Int(outputDimension) {
            var str = ""
            for j in 0..<Int(inputDimension) {
                str += String(self.weightsMatrix[i * Int(inputDimension) + j]);
                str += " "
            }
            print(str)
        }
    }
    
    func printBiasesVector() {
        print("Biases Vector")
        var str = ""
        for i in 0..<Int(outputDimension) {
            str += String(self.weightsMatrix[i as Int]);
            str += " "
        }
        print(str)
    }
}
