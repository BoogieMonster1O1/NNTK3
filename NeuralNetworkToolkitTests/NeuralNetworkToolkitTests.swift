//
//  NeuralNetworkToolkitTests.swift
//  NeuralNetworkToolkitTests
//
//  Created by Shrish Deshpande on 5/2/23.
//

import XCTest
import NeuralNetworkToolkit

final class NeuralNetworkToolkitTests: XCTestCase {

    override func setUpWithError() throws {
    }

    override func tearDownWithError() throws {
    }

    func testCreation() throws {
        let neuralNetwork = NNTKNeuralNetwork(inputDimension: 3, outputDimension: 5)
        neuralNetwork.addHiddenLayer(withOutputDimension: 4)
        neuralNetwork.addHiddenLayer(withOutputDimension: 8)
        neuralNetwork.addHiddenLayer(withOutputDimension: 7)
        neuralNetwork.addHiddenLayer(withOutputDimension: 8)
        XCTAssertEqual(neuralNetwork.layerCount, 5 as UInt, "Wrong layer count")
        neuralNetwork.getLayer(at: 0).assertInOut(3, 4)
        neuralNetwork.getLayer(at: 1).assertInOut(4, 8)
        neuralNetwork.getLayer(at: 2).assertInOut(8, 7)
        neuralNetwork.getLayer(at: 3).assertInOut(7, 8)
        neuralNetwork.getLayer(at: 4).assertInOut(8, 5)
    }

    func testSingleForward() throws {
        let layer = NNTKLayer(activationFunction: NNTKReLUActivationFunction(), inputDimension: 2, outputDimension: 2)
        layer.weightsMatrix[0] = 2
        layer.weightsMatrix[1] = 0
        layer.weightsMatrix[2] = 0
        layer.weightsMatrix[3] = 2
        layer.biasesVector[0] = 0
        layer.biasesVector[1] = 2
        let input: UnsafeMutablePointer<Float> = calloc(2, 4)!.assumingMemoryBound(to: Float.self)
        input[0] = 5
        input[1] = 4
        layer.printBiasesVector()
        layer.printWeightsMatrix()
        let output1 = layer.forward(input)
        XCTAssertEqual(output1[0], 10)
        XCTAssertEqual(output1[1], 10)
        output1.deallocate()
    }
}
