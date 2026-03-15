//
//  Untitled.swift
//  testApp
//
//  Created by Sanjib Sarkar on 3/5/26.
//

import Foundation
import AVFoundation
import Observation

@Observable
final class CameraController {
    let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "com.testApp.camera.sessionQueue")
    
    init() {
        configureSession()
    }

    private func configureSession() {
        sessionQueue.async {
            self.session.beginConfiguration()
            self.session.sessionPreset = .high

            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: device),
                  self.session.canAddInput(input)
            else {
                self.session.commitConfiguration()
                return
            }

            self.session.addInput(input)
            self.session.commitConfiguration()
        }
    }

    func start() {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        
        if status == .authorized {
            sessionQueue.async {
                if !self.session.isRunning {
                    self.session.startRunning()
                }
            }
        } else if status == .notDetermined {
            AVCaptureDevice.requestAccess(for: .video) { granted in
                if granted {
                    self.start()
                }
            }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning {
                self.session.stopRunning()
            }
        }
    }
}
