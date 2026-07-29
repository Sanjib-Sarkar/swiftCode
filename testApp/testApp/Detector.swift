//
//  Detector.swift
//  testApp
//
//  Created by Sanjib Sarkar on 3/16/26.
//

import Vision
import CoreML

struct Detection {
    let box: CGRect
    let label: String
    let confidence: Float
}

final class Detector {
    private var model: VNCoreMLModel?

    init() {
        let config = MLModelConfiguration()
        config.computeUnits = .all // Uses iPhone 12 Neural Engine
        if let coreModel = try? MLModel(contentsOf: Bundle.main.url(forResource: AppConfig.modelName, withExtension: "mlmodelc")!, configuration: config) {
            self.model = try? VNCoreMLModel(for: coreModel)
        }
    }

    func processFrame(_ buffer: CVPixelBuffer, completion: @escaping ([Detection]) -> Void) {
        guard let model = model else { return }

        let request = VNCoreMLRequest(model: model) { request, _ in
            // Pull the raw MultiArray (using the key from your code: var_1227)
            guard let observations = request.results as? [VNCoreMLFeatureValueObservation],
                  let multiArray = observations.first(where: { $0.featureName == "var_1227" })?.featureValue.multiArrayValue else { return }
            
            completion(self.parse(multiArray))
        }
        
        request.imageCropAndScaleOption = .scaleFill
        try? VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up).perform([request])
    }

    private func parse(_ array: MLMultiArray) -> [Detection] {
        var detections = [Detection]()
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        
        // Shape [1, num_attributes, num_detections]
        let numAttributes = array.shape[1].intValue
        let numDetections = array.shape[2].intValue
        
        for i in 0..<numDetections {
            let conf = ptr[4 * numDetections + i] // Assuming 5th attribute is confidence
            if conf > AppConfig.confidenceThreshold {
                let x = CGFloat(ptr[0 * numDetections + i])
                let y = CGFloat(ptr[1 * numDetections + i])
                let w = CGFloat(ptr[2 * numDetections + i])
                let h = CGFloat(ptr[3 * numDetections + i])
                
                // Find best class
                var bestScore: Float = 0
                var classIdx = 0
                for c in 0..<(numAttributes - 4) {
                    let score = ptr[(4 + c) * numDetections + i]
                    if score > bestScore { bestScore = score; classIdx = c }
                }

                detections.append(Detection(
                    box: CGRect(x: x - w/2, y: y - h/2, width: w, height: h),
                    label: AppConfig.classNames[classIdx],
                    confidence: conf
                ))
            }
        }
        return detections
    }
}
