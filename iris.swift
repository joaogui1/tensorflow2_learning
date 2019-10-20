import TensorFlow

import Python
print(Python.version)
let plt = Python.import("matplotlib.pyplot")

import Foundation
import FoundationNetworking

/// Licensed under the Apache License, Version 2.0 (the "License");
/// you may not use this file except in compliance with the License.
/// You may obtain a copy of the License at
///
/// https://www.apache.org/licenses/LICENSE-2.0
///
/// Unless required by applicable law or agreed to in writing, software
/// distributed under the License is distributed on an "AS IS" BASIS,
/// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
/// See the License for the specific language governing permissions and
/// limitations under the License.

/// This file contains some special-case APIs that make parts of the tutorial work even though we
/// do not have general APIs. Soon, we will have general APIs replacing this.

/// Conform `IrisBatch` to `TensorGroup` so that we can load it into a `Dataset`.

/// Initialize an `IrisBatch` dataset from a CSV file.


func download(from sourceString: String, to destinationString: String){
    let source = URL(string: sourceString)!
    let destination = URL(fileURLWithPath: destinationString)
    let data = try! Data.init(contentsOf: source)
    try! data.write(to: destination)
}

download(
    from: "https://raw.githubusercontent.com/tensorflow/swift/master/docs/site/tutorials/TutorialDatasetCSVAPI.swift",
    to: "TutorialDatasetCSVAPI.swift")
let trainDataFilename = "iris_training.csv"
download(from: "http://download.tensorflow.org/data/iris_training.csv", to: trainDataFilename)

let f = Python.open(trainDataFilename)
for _ in 0..<5 {
    print(Python.next(f).strip())
}
f.close()

let featureNames = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
let labelName = "species"
let columnNames = featureNames + [labelName]

print("Features: \(featureNames)")
print("Label: \(labelName)")

let classNames = ["Iris setosa", "Iris versicolor", "Iris virginica"]

let batchSize = 32

/// A batch of examples from the iris dataset.
struct IrisBatch {
    /// [batchSize, featureCount] tensor of features.
    let features: Tensor<Float>

    /// [batchSize] tensor of labels.
    let labels: Tensor<Int32>
}

extension IrisBatch: TensorGroup {
    public static var _typeList: [TensorDataType] = [
        Float.tensorFlowDataType,
        Int32.tensorFlowDataType
    ]
    public static var _unknownShapeList: [TensorShape?] = [nil, nil]
    public var _tensorHandles: [_AnyTensorHandle] {
        fatalError("unimplemented")
    }
    public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?) {
        address!.advanced(by: 0).initialize(to: features.handle._cTensorHandle)
        address!.advanced(by: 1).initialize(to: labels.handle._cTensorHandle)
    }
    public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
        features = Tensor(handle: TensorHandle(_owning: tensorHandles!.advanced(by: 0).pointee))
        labels = Tensor(handle: TensorHandle(_owning: tensorHandles!.advanced(by: 1).pointee))
    }
    public init<C: RandomAccessCollection>(_handles: C) where C.Element: _AnyTensorHandle {
        fatalError("unimplemented")
    }
}

extension Dataset where Element == IrisBatch {
    init(
        contentsOfCSVFile: String, hasHeader: Bool, featureColumns: [Int], labelColumns: [Int]
    ) {
        let np = Python.import("numpy")

        let featuresNp = np.loadtxt(
            contentsOfCSVFile,
            delimiter: ",",
            skiprows: hasHeader ? 1 : 0,
            usecols: featureColumns,
            dtype: Float.numpyScalarTypes.first!)
        guard let featuresTensor = Tensor<Float>(numpy: featuresNp) else {
            // This should never happen, because we construct numpy in such a
            // way that it should be convertible to tensor.
            fatalError("np.loadtxt result can't be converted to Tensor")
        }

        let labelsNp = np.loadtxt(
            contentsOfCSVFile,
            delimiter: ",",
            skiprows: hasHeader ? 1 : 0,
            usecols: labelColumns,
            dtype: Int32.numpyScalarTypes.first!)
        guard let labelsTensor = Tensor<Int32>(numpy: labelsNp) else {
            // This should never happen, because we construct numpy in such a
            // way that it should be convertible to tensor.
            fatalError("np.loadtxt result can't be converted to Tensor")
        }

        self.init(elements: IrisBatch(features: featuresTensor, labels: labelsTensor))
    }
}

/// Sequence doesn't have a non-predicated `first` property, so we define one.
/// TODO: Add this to Swift's stdlib.
extension Sequence where Element == IrisBatch {
    var first: IrisBatch? {
        return first(where: { _ in true })
    }
}


let trainDataset: Dataset<IrisBatch> = Dataset(
    contentsOfCSVFile: trainDataFilename, hasHeader: true,
    featureColumns: [0, 1, 2, 3], labelColumns: [4]
).batched(batchSize)


let firstTrainExamples = trainDataset.first!
let firstTrainFeatures = firstTrainExamples.features
let firstTrainLabels = firstTrainExamples.labels
//print("First batch of features: \(firstTrainFeatures)")
//print("First batch of labels: \(firstTrainLabels)")

//let firstTrainFeaturesTransposed = firstTrainFeatures.transposed()
//let petalLengths = firstTrainFeaturesTransposed[2].scalars
//let sepalLengths = firstTrainFeaturesTransposed[0].scalars

//plt.scatter(petalLengths, sepalLengths, c: firstTrainLabels.array.scalars)
//plt.xlabel("Petal length")
//plt.ylabel("Sepal length")
//plt.show()

import TensorFlow

let hiddenSize: Int = 10
struct IrisModel: Layer {
    var layer1 = Dense<Float>(inputSize: 4, outputSize: hiddenSize, activation: relu)
    var layer2 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer3 = Dense<Float>(inputSize: hiddenSize, outputSize: 3)
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}

var model = IrisModel()

// Apply the model to a batch of features.
let firstTrainPredictions = model(firstTrainFeatures)
let untrainedLogits = model(firstTrainFeatures)
let untrainedLoss = softmaxCrossEntropy(logits: untrainedLogits, labels: firstTrainLabels)
print("Loss test: \(untrainedLoss)")

let optimizer = SGD(for: model, learningRate: 0.01)
let (loss, grads) = model.valueWithGradient { model -> Tensor<Float> in
    let logits = model(firstTrainFeatures)
    return softmaxCrossEntropy(logits: logits, labels: firstTrainLabels)
}
print("Current loss: \(loss)")

let epochCount = 500
var trainAccuracyResults: [Float] = []
var trainLossResults: [Float] = []

func accuracy(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
    return Tensor<Float>(predictions .== truths).mean().scalarized()
}

for epoch in 1...epochCount {
    var epochLoss: Float = 0
    var epochAccuracy: Float = 0
    var batchCount: Int = 0
    for batch in trainDataset {
        let (loss, grad) = model.valueWithGradient { (model: IrisModel) -> Tensor<Float> in
            let logits = model(batch.features)
            return softmaxCrossEntropy(logits: logits, labels: batch.labels)
        }
        optimizer.update(&model.self, along: grad)
        let logits = model(batch.features)
        epochAccuracy += accuracy(predictions: logits.argmax(squeezingAxis: 1), truths: batch.labels)
        epochLoss += loss.scalarized()
        batchCount += 1
    }
    epochAccuracy /= Float(batchCount)
    epochLoss /= Float(batchCount)
    trainAccuracyResults.append(epochAccuracy)
    trainLossResults.append(epochLoss)
    if epoch % 50 == 0 {
        print("Epoch \(epoch): Loss: \(epochLoss), Accuracy: \(epochAccuracy)")
    }
}

let testDataFilename = "iris_test.csv"
download(from: "http://download.tensorflow.org/data/iris_test.csv", to: testDataFilename)

let testDataset: Dataset<IrisBatch> = Dataset(
    contentsOfCSVFile: testDataFilename, hasHeader: true,
    featureColumns: [0, 1, 2, 3], labelColumns: [4]
).batched(batchSize)

// NOTE: With `batchSize = 32` and 30 examples in the test dataset, only one batch will run in the loop.
for testBatch in testDataset {
    let logits = model(testBatch.features)
    let predictions = logits.argmax(squeezingAxis: 1)
    print("Test batch accuracy: \(accuracy(predictions: predictions, truths: testBatch.labels))")
}

let firstTestBatch = testDataset.first!
let firstTestBatchLogits = model(firstTestBatch.features)
let firstTestBatchPredictions = firstTestBatchLogits.argmax(squeezingAxis: 1)

print(firstTestBatchPredictions)
print(firstTestBatch.labels)
