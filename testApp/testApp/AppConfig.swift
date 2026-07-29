//
//  AppConfig.swift
//  testApp
//
//  Created by Sanjib Sarkar on 3/16/26.
//
import AVFoundation

struct AppConfig {
    static let modelName = "wsn_sn_sn_11n480_640_nms"
    static let classNames = ["w", "s", "n", "sn", "ss"]
    static let confidenceThreshold: Float = 0.35
    
    // Hardware Settings
    static let cameraPosition: AVCaptureDevice.Position = .back
    static let resolution: AVCaptureSession.Preset = .vga640x480 // Match your model's input
}
