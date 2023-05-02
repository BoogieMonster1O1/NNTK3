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
