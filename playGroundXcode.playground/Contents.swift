import UIKit
import CoreML
import Vision

var greeting = "Hello, playground"
print(greeting)
print("Hi")

/*

// CONFIGURATION

let targetW: CGFloat = 640  // model' W
let targetH: CGFloat = 480 // MODEL'S H  NOT THE original frame's h
let imageName = "1.png"

struct Detection {
    let box: CGRect
    let confidence: Float
    let classIndex: Int
    let className: String
}

// LOAD the test image

func loadImage(named name: String) -> UIImage?{
    guard let image = UIImage(named: name) else{
        print("Could not find \(name) in Resources.")
        return nil
    }
    let w = image.size.width
    let h = image.size.height
    
    var c = 0
    if let cgImage = image.cgImage{
        c = cgImage.bitsPerPixel / cgImage.bitsPerComponent
    }
    print("Image Loaded: \(name)")
    print("Dimensions: \(Int(w)) w x \(Int(h)) h")
    print("Channels: \(c)")
    return image
}

let rawImg = loadImage(named: imageName)

func applyLetterbox(to image: UIImage, targetSize: CGSize)-> UIImage{
    let scale = min(targetSize.width / image.size.width, targetSize.height/image.size.height)
    
    let newW = image.size.width * scale
    let newH = image.size.height * scale
    
    print("Scale: \(scale)")
    
    // calculate the centering offsets- padding
    
    let xOffset = (targetSize.width - newW) / 2.0
    let yOffset = (targetSize.height - newH) / 2.0
    
    //create a black canvas of the target size
    
    UIGraphicsBeginImageContextWithOptions(targetSize, true, 1.0)
    let context = UIGraphicsGetCurrentContext()
    
    context?.setFillColor(UIColor.black.cgColor)
    context?.fill(CGRect(origin: .zero, size: targetSize))
    
    // Draw the scaled image into the center
    let renderRect = CGRect(x: xOffset, y: yOffset, width: newW, height: newH)
        image.draw(in: renderRect)
    
    let resultImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
    
    print("Letterbox applied: \(Int(newW))x\(Int(newH)) image on \(Int(targetSize.width))x\(Int(targetSize.height)) canvas")
    
    return resultImage ?? image
}

if let myRawImg = rawImg {
    let finalImage = applyLetterbox(to: myRawImg, targetSize: CGSize(width: targetW, height: targetH))
    
    finalImage
}

func getPixelBuffer(from image: UIImage) -> CVPixelBuffer? {
    let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.size.width), Int(image.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
    
    guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
    
    CVPixelBufferLockBaseAddress(buffer, .init(rawValue: 0))
    let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: Int(image.size.width), height: Int(image.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
    
    context?.draw(image.cgImage!, in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
    CVPixelBufferUnlockBaseAddress(buffer, .init(rawValue: 0))
    
    return buffer
}

func nMS(detections: [Detection], iouThreshold: CGFloat = 0.45) -> [Detection] {
    // Sort by confidence (highest first)
    var sortedDetections = detections.sorted { $0.confidence > $1.confidence }
    var keptDetections = [Detection]()
    
    while !sortedDetections.isEmpty {
        let best = sortedDetections.removeFirst()
        keptDetections.append(best)
        
        // Remove any other boxes that overlap too much with the 'best' box
        sortedDetections.removeAll { other in
            let intersection = best.box.intersection(other.box)
            let intersectionArea = intersection.width * intersection.height
            let unionArea = (best.box.width * best.box.height) + (other.box.width * other.box.height) - intersectionArea
            let iou = intersectionArea / unionArea
            
            // If they overlap more than the threshold, they are likely the same object
            return iou > iouThreshold && best.classIndex == other.classIndex
        }
    }
    return keptDetections
}

func postProcess(multiArray: MLMultiArray, confidenceThreshold: Float = 0.5) ->[Detection] {
    let numAttributes = multiArray.shape[1].intValue
    let numAnchors = multiArray.shape[2].intValue
    let numClasses = numAttributes - 4
    let classes = ["w", "s", "n", "sn", "ss"]
    
    // Safety check: Ensure the model is actually giving us Float32
    guard multiArray.dataType == .float32 else {
        print("Error: MultiArray is not Float32. Type is: \(multiArray.dataType.rawValue)")
        return [Detection]()
    }

    // FIX: Using assumingMemoryBound
    let pointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
    var allDetections = [Detection]()
    //print("--- Dynamic Detections ---")
    
    for i in 0..<numAnchors {
        var highestScore: Float = 0
        var bestClassIndex = -1
        
        for classIdx in 0..<numClasses {
            let scoreIndex = (4 + classIdx) * numAnchors + i
            let score = pointer[scoreIndex]
            
            if score > highestScore {
                highestScore = score
                bestClassIndex = classIdx
            }
        }
        
        if highestScore > confidenceThreshold {
                    // YOLOv11 usually gives x_center, y_center, width, height
                    let w = CGFloat(pointer[2 * numAnchors + i])
                    let h = CGFloat(pointer[3 * numAnchors + i])
                    let x = CGFloat(pointer[0 * numAnchors + i]) - (w / 2)
                    let y = CGFloat(pointer[1 * numAnchors + i]) - (h / 2)
                    
                    let rect = CGRect(x: x, y: y, width: w, height: h)
                    let det = Detection(box: rect, confidence: highestScore, classIndex: bestClassIndex, className: classes[bestClassIndex])
                    allDetections.append(det)
        }
    }
    
    // RUN NMS TO CLEAN UP DUPLICATES
        let finalDetections = nMS(detections: allDetections)
        
        print("--- Detections (\(finalDetections.count)) ---")
        for d in finalDetections {
            print("\(d.className) (\(Int(d.confidence * 100))%) at \(d.box)")
        }
    return finalDetections
}



func printModelPredictions(_ prediction: MLFeatureProvider, classNames: [String]? = nil) {
    print("\n=== Model Predictions NO NMS ===")
    var detections: [Detection] = []
    
    if let confidenceMultiArray = prediction.featureValue(for: "confidence")?.multiArrayValue,
       let coordinatesMultiArray = prediction.featureValue(for: "coordinates")? .multiArrayValue{
        let shape = confidenceMultiArray.shape.map { $0.intValue }
        let coordShape = coordinatesMultiArray.shape.map{$0.intValue}
        print("Confidence shape: \(shape)")
        print("Coordinates shape: \(coordShape)")
        
        if shape.count == 1 {
            let numClasses = shape[0]
            var maxConf = 0.0
            var maxClass = 0
            
            for i in 0..<numClasses {
                let conf = confidenceMultiArray[i].doubleValue
                if conf > maxConf {
                    maxConf = conf
                    maxClass = i
                }
            }
            
            let className = classNames?[maxClass] ?? "Class \(maxClass)"
            print("\nTop prediction: \(className) with confidence \(String(format: "%.4f", maxConf))")
            
        } else if shape.count == 2 {
            //print("ElseIf shape = 2")
            let numDetections = shape[0]
            let numClasses = shape[1]
            let numCoords = coordShape.count == 2 ? coordShape[1] : 4
            
            print("\nDetections:")
            for det in 0..<numDetections {
                var maxConf = 0.0
                var maxClass = 0
                
                for cls in 0..<numClasses {
                    let index = det * numClasses + cls
                    let conf = confidenceMultiArray[index].doubleValue
                    if conf > maxConf {
                        maxConf = conf
                        maxClass = cls
                    }
                }
                
                if maxConf > 0.80 {
                    let className = classNames?[maxClass] ?? "Class \(maxClass)"
                    var coords: [Double] = []
                    for c in 0..<numCoords {
                        let coordIndex = det * numCoords + c
                        coords.append(coordinatesMultiArray[coordIndex].doubleValue)
                    }
                    if coords.count >= 4 {
                        // normalized → pixel
                        let pixelX = CGFloat(coords[0]) * targetW
                        let pixelY = CGFloat(coords[1]) * targetH
                        let pixelW = CGFloat(coords[2]) * targetW
                        let pixelH = CGFloat(coords[3]) * targetH
                        // center → top-left
                         let rect = CGRect(
                            x: pixelX - pixelW / 2,
                            y: pixelY - pixelH / 2,
                            width: pixelW,
                            height: pixelH
                            )
                        let detection = Detection(
                            box: rect,
                            confidence: Float(maxConf),
                            classIndex: maxClass,
                            className: className
                            )
                        detections.append(detection)
                        
                        print("Det \(det): \(className), Confidence \(String(format: "%.4f", maxConf)), x: \(String(format: "%.1f", pixelX)), y: \(String(format: "%.1f", pixelY)), w: \(String(format: "%.1f", pixelW)), h: \(String(format: "%.1f", pixelH))")
                    }
                }
                
            }
        }
    }
    //print("Detection \(detections)")
    print("========================\n")
}



let config = MLModelConfiguration()
config.computeUnits = .all
let allResources = Bundle.main.urls(forResourcesWithExtension: nil, subdirectory:nil) ?? []
print("Files detected in Resources: \(allResources.map { $0.lastPathComponent })")
do{
    guard let modelURL=Bundle.main.url(forResource: "wsn_sn_sn_11n480_640_nms", withExtension:"mlmodelc") else{ print("Check: Is the filename spelled exactly right? Is it in the Resources sidebar?")
        fatalError()
    }
    let model = try MLModel(contentsOf: modelURL, configuration: config)
    print("Success! Model loaded directly from its compiled state.")
    
    //check input names
    let inputNames = model.modelDescription.inputDescriptionsByName.keys
    print("Model expected these inputs: \(inputNames)")
    
    if let rawImg = loadImage(named: imageName) {
            let boxedImg = applyLetterbox(to: rawImg, targetSize: CGSize(width: targetW, height: targetH))
            
            if let buffer = getPixelBuffer(from: boxedImg) {
                let input = try MLDictionaryFeatureProvider(dictionary: ["image": buffer])
                let prediction = try model.prediction(from: input)
                
                print("Inference Complete!")
                print("Outputs: \(prediction.featureNames)")
                
                if let multiArray = prediction.featureValue(for: "var_1227")?.multiArrayValue {
                    print("MultiArray Shape: \(multiArray.shape)")
                    print("Data Type: \(multiArray.dataType.rawValue)") // 65536 = Float32, 131072 = Float16
                }
                
                if let outputMultiArray = prediction.featureValue(for: "var_1227")?.multiArrayValue {
                    postProcess(multiArray: outputMultiArray, confidenceThreshold: 0.80)
                }else {
                    // Model has confidence/coordinates outputs - use standard function
                    let classNames = ["w", "s", "n", "sn", "ss"]
                    printModelPredictions(prediction, classNames: classNames)
                }
            }
        }
}catch{
    print("CoreML Error:\(error.localizedDescription)")
}

 */


/*

class Detector {
    
    struct Detection {
        let box: CGRect
        let confidence: Float
        let classIndex: Int
        let className: String
    }
    
    // Member variables (like a .h file)
    private var model: MLModel
    private var targetW: CGFloat
    private var targetH: CGFloat
    private var scale: CGFloat = 0
    private var xOffset: CGFloat = 0
    private var yOffset: CGFloat = 0
    private var classNames: [String] = []
    
    init(modelPath: URL, targetW: CGFloat, targetH: CGFloat) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        self.model = try MLModel(contentsOf: modelPath, configuration: config)
        self.targetW = targetW
        self.targetH = targetH
        self.classNames = ["w", "s", "n", "sn", "ss"]
        
    }
    
    // --- Implementation (like a .cpp file) ---
    
    func inference(imagePath: String, confidenceThreshold: Float = 0.5, show: Bool = true) -> UIImage? {
        
        let preProcessStart = CFAbsoluteTimeGetCurrent()
        // 1. Load
        guard let rawImg = self.loadImage(named: imagePath) else { return nil }
        
        // 2. Pre-process (Updates scale and offsets internally)
        let boxedImg = self.applyLetterbox(to: rawImg, targetSize: CGSize(width: self.targetW, height: self.targetH))
        
        guard let buffer = self.getPixelBuffer(from: boxedImg) else { return nil }
        let preProcessTime = (CFAbsoluteTimeGetCurrent() - preProcessStart) * 1000
        
        do {
            // 3. Predict
            let inferenceStart = CFAbsoluteTimeGetCurrent()
            
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": buffer])
            let prediction = try self.model.prediction(from: input)
            
            let inferenceTime = (CFAbsoluteTimeGetCurrent() - inferenceStart) * 1000
            
            
            //guard let outputArray = prediction.featureValue(for: "var_1227")?.multiArrayValue else { return nil }
            
            let postProcessStart = CFAbsoluteTimeGetCurrent()
            // 4. Post-process (NMS)
            let modelDetections: [Detection]
            if let outputMultiArray = prediction.featureValue(for: "var_1227")?.multiArrayValue {
                modelDetections = self.postProcess(multiArray: outputMultiArray, confidenceThreshold: confidenceThreshold)
            }else {
                // Model has confidence/coordinates outputs - use standard function
                let classNames = ["w", "s", "n", "sn", "ss"]
                modelDetections = self.postProcessNoNms(prediction,confidenceThreshold: confidenceThreshold,classNames: classNames)
                //let modelDetections = self.postProcess(multiArray: outputArray, confidenceThreshold: confidenceThreshold)
            }
            // 5. Map to Original Coordinates (Reuses cached scale/offsets)
            let finalResults = self.mapToOriginal(detections: modelDetections)
            let postProcessTime = (CFAbsoluteTimeGetCurrent() - postProcessStart) * 1000
            
            //print("\n--- Timing Benchmarks ---")
            print("Preprocessing:  \(String(format: "%.2f", preProcessTime)) ms")
            print("Inference:      \(String(format: "%.2f", inferenceTime)) ms")
            print("Post-processing: \(String(format: "%.2f", postProcessTime)) ms")
            print("Total Pipeline: \(String(format: "%.2f", preProcessTime + inferenceTime + postProcessTime)) ms\n")
            
            
            // 6. Visualization
            if show {
                return self.drawDetections(on: rawImg, detections: finalResults)
            }
            
            return rawImg
            
        } catch {
            print("Inference Error: \(error)")
            return nil
        }
    }
    
    
    func loadImage(named name: String) -> UIImage? {
        guard let image = UIImage(named: name) else {
            print("Could not find \(name) in Resources.")
            return nil
        }
        let w = image.size.width
        let h = image.size.height
        //var c = 0
        //if let cgImage = image.cgImage {
        //  c = cgImage.bitsPerPixel / cgImage.bitsPerComponent
        //}
        //print("Image Loaded: \(name)\nDimensions: \(Int(w)) w x \(Int(h)) h\nChannels: \(c)")
        return image
    }
    
    func applyLetterbox(to image: UIImage, targetSize: CGSize) -> UIImage {
        // Calculate and ASSIGN to member variables for reuse
        self.scale = min(targetSize.width / image.size.width, targetSize.height / image.size.height)
        let newW = image.size.width * self.scale
        let newH = image.size.height * self.scale
        
        self.xOffset = (targetSize.width - newW) / 2.0
        self.yOffset = (targetSize.height - newH) / 2.0
        
        UIGraphicsBeginImageContextWithOptions(targetSize, true, 1.0)
        let context = UIGraphicsGetCurrentContext()
        context?.setFillColor(UIColor.black.cgColor)
        context?.fill(CGRect(origin: .zero, size: targetSize))
        
        let renderRect = CGRect(x: self.xOffset, y: self.yOffset, width: newW, height: newH)
        image.draw(in: renderRect)
        
        let resultImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        //print("Scale: \(self.scale)")
        //print("Letterbox applied: \(Int(newW))x\(Int(newH)) image on \(Int(targetSize.width))x\(Int(targetSize.height)) canvas")
        return resultImage ?? image
    }
    
    func mapToOriginal(detections: [Detection]) -> [Detection] {
        // Reuses self.scale, self.xOffset, and self.yOffset automatically
        return detections.map { det in
            let oldBox = det.box
            let newX = (oldBox.origin.x - self.xOffset) / self.scale
            let newY = (oldBox.origin.y - self.yOffset) / self.scale
            let newW = oldBox.width / self.scale
            let newH = oldBox.height / self.scale
            
            return Detection(
                box: CGRect(x: newX, y: newY, width: newW, height: newH),
                confidence: det.confidence,
                classIndex: det.classIndex,
                className: det.className
            )
        }
    }
    
    func getPixelBuffer(from image: UIImage) -> CVPixelBuffer? {
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue, kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, Int(image.size.width), Int(image.size.height), kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(buffer, .init(rawValue: 0))
        let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: Int(image.size.width), height: Int(image.size.height), bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)
        
        // Fix for coordinate flip
        //context?.translateBy(x: 0, y: image.size.height)
        //context?.scaleBy(x: 1.0, y: -1.0)
        
        context?.draw(image.cgImage!, in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
        
        
        
        CVPixelBufferUnlockBaseAddress(buffer, .init(rawValue: 0))
        return buffer
    }
    
    func nMS(detections: [Detection], iouThreshold: CGFloat = 0.45) -> [Detection] {
        var sortedDetections = detections.sorted { $0.confidence > $1.confidence }
        var keptDetections = [Detection]()
        while !sortedDetections.isEmpty {
            let best = sortedDetections.removeFirst()
            keptDetections.append(best)
            sortedDetections.removeAll { other in
                let intersection = best.box.intersection(other.box)
                let intersectionArea = intersection.width * intersection.height
                let unionArea = (best.box.width * best.box.height) + (other.box.width * other.box.height) - intersectionArea
                return (intersectionArea / unionArea) > iouThreshold && best.classIndex == other.classIndex
            }
        }
        return keptDetections
    }
    
    func postProcess(multiArray: MLMultiArray, confidenceThreshold: Float = 0.5) -> [Detection] {
        let numAttributes = multiArray.shape[1].intValue
        let numAnchors = multiArray.shape[2].intValue
        let numClasses = numAttributes - 4
        let classes = ["w", "s", "n", "sn", "ss"]
        
        guard multiArray.dataType == .float32 else { return [] }
        let pointer = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
        var allDetections = [Detection]()
        
        for i in 0..<numAnchors {
            var highestScore: Float = 0
            var bestClassIndex = -1
            for classIdx in 0..<numClasses {
                let scoreIndex = (4 + classIdx) * numAnchors + i
                let score = pointer[scoreIndex]
                if score > highestScore {
                    highestScore = score
                    bestClassIndex = classIdx
                }
            }
            if highestScore > confidenceThreshold {
                let w = CGFloat(pointer[2 * numAnchors + i])
                let h = CGFloat(pointer[3 * numAnchors + i])
                let x = CGFloat(pointer[0 * numAnchors + i]) - (w / 2)
                let y = CGFloat(pointer[1 * numAnchors + i]) - (h / 2)
                allDetections.append(Detection(box: CGRect(x: x, y: y, width: w, height: h), confidence: highestScore, classIndex: bestClassIndex, className: classes[bestClassIndex]))
            }
        }
        if allDetections.count > 55 {
            allDetections = Array(allDetections.sorted { $0.confidence > $1.confidence }.prefix(300))
        }
        
        return nMS(detections: allDetections)
    }
    
    func drawDetections(on image: UIImage, detections: [Detection]) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(image.size, false, 0.0)
        image.draw(at: .zero)
        let context = UIGraphicsGetCurrentContext()
        
        for det in detections {
            context?.setLineWidth(4.0)
            context?.setStrokeColor(UIColor.green.cgColor)
            context?.stroke(det.box)
            
            let text = "\(det.className) \(Int(det.confidence * 100))%"
            let attributes: [NSAttributedString.Key: Any] = [
                .font: UIFont.boldSystemFont(ofSize: 48),
                .foregroundColor: UIColor.green,
                .backgroundColor: UIColor.black.withAlphaComponent(0.5)
            ]
            text.draw(at: CGPoint(x: det.box.origin.x, y: det.box.origin.y - 55), withAttributes: attributes)
        }
        
        let result = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return result ?? image
    }
    
    func postProcessNoNms(_ prediction: MLFeatureProvider, confidenceThreshold: Float = 0.25, classNames: [String]? = nil) -> [Detection]  {
        print("\n=== Model Predictions NO NMS ===")
        var detections: [Detection] = []
        
        if let confidenceMultiArray = prediction.featureValue(for: "confidence")?.multiArrayValue,
           let coordinatesMultiArray = prediction.featureValue(for: "coordinates")? .multiArrayValue{
            let shape = confidenceMultiArray.shape.map { $0.intValue }
            let coordShape = coordinatesMultiArray.shape.map{$0.intValue}
            //print("Confidence shape: \(shape)")
            //print("Coordinates shape: \(coordShape)")
            
            if shape.count == 1 {
                let numClasses = shape[0]
                var maxConf = 0.0
                var maxClass = 0
                
                for i in 0..<numClasses {
                    let conf = confidenceMultiArray[i].doubleValue
                    if conf > maxConf {
                        maxConf = conf
                        maxClass = i
                    }
                }
                
                let className = classNames?[maxClass] ?? "Class \(maxClass)"
                print("\nTop prediction: \(className) with confidence \(String(format: "%.4f", maxConf))")
                
            } else if shape.count == 2 {
                //print("ElseIf shape = 2")
                let numDetections = shape[0]
                let numClasses = shape[1]
                let numCoords = coordShape.count == 2 ? coordShape[1] : 4
                
                print("\nDetections:")
                for det in 0..<numDetections {
                    var maxConf = 0.0
                    var maxClass = 0
                    
                    for cls in 0..<numClasses {
                        let index = det * numClasses + cls
                        let conf = confidenceMultiArray[index].doubleValue
                        if conf > maxConf {
                            maxConf = conf
                            maxClass = cls
                        }
                    }
                    
                    if maxConf > Double(confidenceThreshold){
                        let className = classNames?[maxClass] ?? "Class \(maxClass)"
                        var coords: [Double] = []
                        for c in 0..<numCoords {
                            let coordIndex = det * numCoords + c
                            coords.append(coordinatesMultiArray[coordIndex].doubleValue)
                        }
                        if coords.count >= 4 {
                            // normalized → pixel
                            let pixelX = CGFloat(coords[0]) * targetW
                            let pixelY = CGFloat(coords[1]) * targetH
                            let pixelW = CGFloat(coords[2]) * targetW
                            let pixelH = CGFloat(coords[3]) * targetH
                            // center → top-left
                            let rect = CGRect(
                                x: pixelX - pixelW / 2,
                                y: pixelY - pixelH / 2,
                                width: pixelW,
                                height: pixelH
                            )
                            let detection = Detection(
                                box: rect,
                                confidence: Float(maxConf),
                                classIndex: maxClass,
                                className: className
                            )
                            detections.append(detection)
                            
                            print("Det \(det): \(className), Confidence \(String(format: "%.4f", maxConf)), x: \(String(format: "%.1f", pixelX)), y: \(String(format: "%.1f", pixelY)), w: \(String(format: "%.1f", pixelW)), h: \(String(format: "%.1f", pixelH))")
                        }
                    }
                }
            }
        }
        return detections
    }
}
        

guard let modelURL = Bundle.main.url(forResource: "wsn_sn_sn_11n480_640_nms", withExtension: "mlmodelc") else {
    fatalError("Model file not found.")
}

do {
    // 2. Initialize the class (Constructor)
    let detector = try Detector(modelPath: modelURL, targetW: 640, targetH: 480)
    
    /*
    // Run full inference in one line
    if let resultImage = detector.inference(imagePath: "1.png", confidenceThreshold: 0.75, show: true) {
        // Display in Playground
        resultImage
    }*/
    
    // loop through a folder
    let extensions = ["png", "jpg", "jpeg"]
    let imageURLs = Bundle.main.urls(forResourcesWithExtension: nil, subdirectory: "testImages")?.filter { url in
            extensions.contains(url.pathExtension.lowercased())
        } ?? []
    print(" Found \(imageURLs.count) images in 'testImages' folder.")
    
    /*
    for url in imageURLs {
            let fileName = url.lastPathComponent
            //print("\n Processing: \(fileName)")
            
            _ = detector.inference(imagePath: "testImages/\(fileName)", confidenceThreshold: 0.75, show: false)
        
            // Pass the path relative to the subdirectory or the full path
            //if let resultImage = detector.inference(imagePath: "testImages/\(fileName)", confidenceThreshold: 0.75, show: //false) {
                // In Playgrounds, this will show the last image processed in the sidebar,
                //resultImage
            //}
        }
     */
    
    for url in imageURLs {
        autoreleasepool { // This forces the "raw image" memory to clear immediately
            let fileName = url.lastPathComponent
            print("\n Processing: \(fileName)")
            
            // Use the underscore to prevent the Playground from capturing the result
            _ = detector.inference(imagePath: "testImages/\(fileName)", confidenceThreshold: 0.35, show: true)
        }
    }
    print("*************DONE*****************")
    
} catch {
    print("Initialization failed: \(error)")
}

 */


 print("DetecotVision")
 
class DetectorVision {
    struct Detection {
        let box: CGRect
        let confidence: Float
        let classIndex: Int
        let className: String
    }
    private var model: VNCoreMLModel
    private var targetW: CGFloat
    private var targetH: CGFloat
    private var classNames : [String] = []
    
    // Vision handles scaling automatically, but we store these to map boxes back
    private var currentImageSize: CGSize = .zero
    
    init(modelPath: URL, targetW: CGFloat, targetH: CGFloat) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let coreMLModel = try MLModel(contentsOf: modelPath, configuration: config)
        
        // Wrap the CoreML model in a Vision model
        self.model = try VNCoreMLModel(for: coreMLModel)
        
        self.targetW = targetW
        self.targetH = targetH
        self.classNames = ["w", "s", "n", "sn", "ss"]
    }
    
    func inference(url: URL, confidenceThreshold: Float = 0.5, show: Bool = true) -> UIImage? {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        var detections = [Detection]()
        
        let startPrep = CFAbsoluteTimeGetCurrent()
        
        guard let sourceImage = UIImage(contentsOfFile: url.path),
              let cgImage = sourceImage.cgImage else { return nil }
        
        self.currentImageSize = sourceImage.size
        
        let prepTime = (CFAbsoluteTimeGetCurrent() - startPrep) * 1000
        
        var inferenceTime: Double = 0
        var postProcessTime: Double = 0
        
        let request = VNCoreMLRequest(model: self.model) { request, error in
            
            if let error = error {
                print("Vision request error: \(error)")
                return
            }
            
            let startInf = CFAbsoluteTimeGetCurrent()
            let detectionResults: [Detection]
            
            if let observations = request.results as? [VNCoreMLFeatureValueObservation] {
                
                if let multiArrayObs = observations.first(where: { $0.featureValue.multiArrayValue != nil }),
                   let multiArray = multiArrayObs.featureValue.multiArrayValue {
                    
                    let ppStart = CFAbsoluteTimeGetCurrent()
                    detectionResults = self.postProcess(
                        multiArray: multiArray,
                        confidenceThreshold: confidenceThreshold
                    )
                    postProcessTime = (CFAbsoluteTimeGetCurrent() - ppStart) * 1000
                    
                } else {
                    
                    var featureDict: [String: MLFeatureValue] = [:]
                    for obs in observations {
                        let name = obs.featureName
                        featureDict[name] = obs.featureValue
                    }
                    
                    if let prediction = try? MLDictionaryFeatureProvider(dictionary: featureDict) {
                        
                        let ppStart = CFAbsoluteTimeGetCurrent()
                        detectionResults = self.postProcessNoNms(
                            prediction,
                            confidenceThreshold: confidenceThreshold,
                            classNames: self.classNames
                        )
                        postProcessTime = (CFAbsoluteTimeGetCurrent() - ppStart) * 1000
                        
                    } else {
                        detectionResults = []
                    }
                }
                
            } else {
                detectionResults = []
            }
            
            inferenceTime = (CFAbsoluteTimeGetCurrent() - startInf) * 1000
            detections = detectionResults
        }
        
        request.imageCropAndScaleOption = .scaleFit
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        
        do {
            try handler.perform([request])
        } catch {
            print("Vision Error: \(error)")
            return nil
        }
        
        let totalTime = (CFAbsoluteTimeGetCurrent() - startTime) * 1000
        
        print("Image Prep: \(String(format: "%.2f", prepTime)) ms")
        print("Inf + Post: \(String(format: "%.2f", inferenceTime)) ms")
        print("Pipeline Total: \(String(format: "%.2f", totalTime)) ms")
        print()
        
        if show {
            return self.drawDetections(on: sourceImage, detections: detections)
        }
        
        return nil
    }
    
    // Optimized Post-process using direct pointer access (Unsafe)
    func postProcess(multiArray: MLMultiArray, confidenceThreshold: Float = 0.5) -> [Detection] {
        let numAttributes = multiArray.shape[1].intValue // 85 (4 box + 81 classes)
        let numAnchors = multiArray.shape[2].intValue    // 8400
        let numClasses = numAttributes - 4
        let classes = ["w", "s", "n", "sn", "ss"]
        
        var allDetections = [Detection]()
        // Direct pointer access bypasses Swift bounds checking
        let ptr = multiArray.dataPointer.assumingMemoryBound(to: Float.self)
        let scale = min(targetW / currentImageSize.width,
                        targetH / currentImageSize.height)
        
        let scaledWidth = currentImageSize.width * scale
        let scaledHeight = currentImageSize.height * scale
        
        let padX = (targetW - scaledWidth) / 2
        let padY = (targetH - scaledHeight) / 2
        
        for i in 0..<numAnchors {
            var highestScore: Float = 0
            var bestClassIndex = -1
            
            for classIdx in 0..<numClasses {
                let score = ptr[(4 + classIdx) * numAnchors + i]
                if score > highestScore {
                    highestScore = score
                    bestClassIndex = classIdx
                }
            }
            
            if highestScore > confidenceThreshold {
                let centerX = CGFloat(ptr[0 * numAnchors + i])
                let centerY = CGFloat(ptr[1 * numAnchors + i])
                let w = CGFloat(ptr[2 * numAnchors + i])
                let h = CGFloat(ptr[3 * numAnchors + i])
                
                // Convert Normalized Model Coordinates to Pixel Coordinates
                // Vision outputs are relative to the input image size
                let x = (centerX - w/2 - padX) / scale
                let y = (centerY - h/2 - padY) / scale
                let width = w / scale
                let height = h / scale
                let rect = CGRect(x: x, y: y,
                                  width: width, height: height)
                allDetections.append(Detection(box: rect, confidence: highestScore, classIndex: bestClassIndex, className: classes[bestClassIndex]))
            }
            
        }
        if allDetections.count > 55 {
            allDetections = Array(allDetections.sorted { $0.confidence > $1.confidence }.prefix(300))
        }
        return nMS(detections: allDetections)
    }
    
    func nMS(detections: [Detection], iouThreshold: CGFloat = 0.45) -> [Detection] {
        if detections.isEmpty { return [] }
        var sortedDetections = detections.sorted { $0.confidence > $1.confidence }
        var keptDetections = [Detection]()
        var suppressed = Array(repeating: false, count: sortedDetections.count)
        
        for i in 0..<sortedDetections.count {
            if suppressed[i] { continue }
            let best = sortedDetections[i]
            keptDetections.append(best)
            
            for j in (i + 1)..<sortedDetections.count {
                if suppressed[j] { continue }
                let other = sortedDetections[j]
                if best.classIndex != other.classIndex { continue }
                let intersection = best.box.intersection(other.box)
                let intersectionArea = intersection.width * intersection.height
                if intersectionArea <= 0 { continue }
                let unionArea =
                (best.box.width * best.box.height) +
                (other.box.width * other.box.height) -
                intersectionArea
                if (intersectionArea / unionArea) > iouThreshold {
                    suppressed[j] = true
                }
            }
        }
        return keptDetections
    }
    
    func drawDetections(on image: UIImage, detections: [Detection]) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(image.size, false, 0.0)
        image.draw(at: .zero)
        let context = UIGraphicsGetCurrentContext()
        for det in detections {
            context?.setLineWidth(max(image.size.width/200, 2.0))
            context?.setStrokeColor(UIColor.green.cgColor)
            context?.stroke(det.box)
            //let label = "\(det.className) \(String(format: "%.2f", det.confidence))"
            let label = "\(det.className)"
            
            let fontSize = max(image.size.width / 40, 14)
            let font = UIFont.boldSystemFont(ofSize: fontSize)
            
            let attributes: [NSAttributedString.Key: Any] = [
                .font: font,
                .foregroundColor: UIColor.white]
            let textSize = label.size(withAttributes: attributes)
            
            // Position text above bounding box
            var textRect = CGRect(
                x: det.box.origin.x,
                y: det.box.origin.y - textSize.height - 4,
                width: textSize.width + 8,
                height: textSize.height + 4
            )
            // Prevent label from going outside top boundary
            if textRect.origin.y < 0 {
                textRect.origin.y = det.box.origin.y + 2
            }
            
            // Draw background rectangle
            //context?.setFillColor(UIColor.green.cgColor)
            //context?.fill(textRect)
            
            // Draw text
            let textPoint = CGPoint(
                x: textRect.origin.x + 4,
                y: textRect.origin.y + 2
            )
            label.draw(at: textPoint, withAttributes: attributes)
        }
        let result = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return result ?? image
    }
    
    func postProcessNoNms(_ prediction: MLFeatureProvider, confidenceThreshold: Float = 0.25, classNames: [String]? = nil) -> [Detection]  {
        print("\n=== Model Predictions NO NMS ===")
        var detections: [Detection] = []
        
        if let confidenceMultiArray = prediction.featureValue(for: "confidence")?.multiArrayValue,
           let coordinatesMultiArray = prediction.featureValue(for: "coordinates")? .multiArrayValue{
            let shape = confidenceMultiArray.shape.map { $0.intValue }
            let coordShape = coordinatesMultiArray.shape.map{$0.intValue}
            //print("Confidence shape: \(shape)")
            //print("Coordinates shape: \(coordShape)")
            if shape.count == 1 {
                let numClasses = shape[0]
                var maxConf = 0.0
                var maxClass = 0
                
                for i in 0..<numClasses {
                    let conf = confidenceMultiArray[i].doubleValue
                    if conf > maxConf {
                        maxConf = conf
                        maxClass = i
                    }
                }
                let className = classNames?[maxClass] ?? "Class \(maxClass)"
                print("\nTop prediction: \(className) with confidence \(String(format: "%.4f", maxConf))")
            }else if shape.count == 2 {
                //print("ElseIf shape = 2")
                let numDetections = shape[0]
                let numClasses = shape[1]
                let numCoords = coordShape.count == 2 ? coordShape[1] : 4
                
                print("\nDetections:")
                for det in 0..<numDetections {
                    var maxConf = 0.0
                    var maxClass = 0
                    
                    for cls in 0..<numClasses {
                        let index = det * numClasses + cls
                        let conf = confidenceMultiArray[index].doubleValue
                        if conf > maxConf {
                            maxConf = conf
                            maxClass = cls
                        }
                    }
                    
                    if maxConf > Double(confidenceThreshold){
                        let className = classNames?[maxClass] ?? "Class \(maxClass)"
                        var coords: [Double] = []
                        for c in 0..<numCoords {
                            let coordIndex = det * numCoords + c
                            coords.append(coordinatesMultiArray[coordIndex].doubleValue)
                        }
                        if coords.count >= 4 {
                            // normalized → pixel
                            let pixelX = CGFloat(coords[0]) * targetW
                            let pixelY = CGFloat(coords[1]) * targetH
                            let pixelW = CGFloat(coords[2]) * targetW
                            let pixelH = CGFloat(coords[3]) * targetH
                            // center → top-left
                            let rect = CGRect(
                                x: pixelX - pixelW / 2,
                                y: pixelY - pixelH / 2,
                                width: pixelW,
                                height: pixelH
                            )
                            let detection = Detection(
                                box: rect,
                                confidence: Float(maxConf),
                                classIndex: maxClass,
                                className: className
                            )
                            detections.append(detection)
                            
                            print("Det \(det): \(className), Confidence \(String(format: "%.4f", maxConf)), x: \(String(format: "%.1f", pixelX)), y: \(String(format: "%.1f", pixelY)), w: \(String(format: "%.1f", pixelW)), h: \(String(format: "%.1f", pixelH))")
                        }
                    }
                }
            }
        }
        return detections
    }
}

guard let modelURL = Bundle.main.url(forResource: "wsn_sn_sn_11n480_640_nms", withExtension: "mlmodelc") else {
    fatalError("Model file not found.")
}

do {
    // 1. Initialize with Vision-ready Detector
    let detector = try DetectorVision(modelPath: modelURL, targetW: 640, targetH: 480)
    // 2. Get images from the subdirectory
    let extensions = ["png", "jpg", "jpeg"]
    let folderName = "testImages"
    // Bundle.main.urls specifically for the subdirectory
    let imageURLs = Bundle.main.urls(forResourcesWithExtension: nil, subdirectory: folderName)?.filter { url in
        extensions.contains(url.pathExtension.lowercased())
    } ?? []
    
    print("Found \(imageURLs.count) images. Starting Vision Inference...")
    
    // 3. Loop with memory management
    for url in imageURLs {
        autoreleasepool {
            let fileName = url.lastPathComponent
            print("Processing: \(fileName)")
            // pass the full URL now for the VNImageRequestHandler to handle efficiently
            // 'show: false' keeps the result sidebar clean and maximizes performance
            _ = detector.inference(url: url, confidenceThreshold: 0.50, show: true)
        }
    }
    print("\n----------- DONE ------------")
    
} catch {
    print("Initialization failed: \(error)")
}
 


