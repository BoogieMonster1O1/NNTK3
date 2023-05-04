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
        let weights: UnsafeMutablePointer<Float> = malloc(4 * MemoryLayout<Float>.stride)!.assumingMemoryBound(to: Float.self)
        let biases: UnsafeMutablePointer<Float> = malloc(2 * MemoryLayout<Float>.stride)!.assumingMemoryBound(to: Float.self)
        weights[0] = 2
        weights[1] = 0
        weights[2] = 0
        weights[3] = 2
        biases[0] = 0
        biases[1] = 2
        let layer = NNTKLayer(weights: NSMutableData(bytesNoCopy: weights, length: 16, freeWhenDone: true), biases: NSMutableData(bytesNoCopy: biases, length: 8, freeWhenDone: true), activationFunction: NNTKReLUActivationFunction(), inputDimension: 2, outputDimension: 2)
        let input: UnsafeMutablePointer<Float> = malloc(2 * MemoryLayout<Float>.stride)!.assumingMemoryBound(to: Float.self)
        input[0] = 5
        input[1] = 4
        layer.printBiasesVector()
        layer.printWeightsMatrix()
        let inputData = Data(bytesNoCopy: input, count: 8, deallocator: .free)
        let data = layer.forward(inputData).bytes.assumingMemoryBound(to: Float.self)
        print(data.debugDescription)
        XCTAssertEqual(data[0], 10)
        XCTAssertEqual(data[1], 10)
    }
}
