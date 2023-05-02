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
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testExample() throws {
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

    func testPerformanceExample() throws {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }
}
