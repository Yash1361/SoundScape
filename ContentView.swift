import UIKit
import AVFoundation
import Vision

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    var captureSession: AVCaptureSession!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var objectDetectionRequest: VNCoreMLRequest!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        setupObjectDetection()
    }
    
    func setupCamera() {
        // Set up camera capture session
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .high
        
        guard let videoCaptureDevice = AVCaptureDevice.default(for: .video) else { return }
        let videoInput: AVCaptureDeviceInput
        
        do {
            videoInput = try AVCaptureDeviceInput(device: videoCaptureDevice)
        } catch {
            return
        }
        
        if (captureSession.canAddInput(videoInput)) {
            captureSession.addInput(videoInput)
        } else {
            return
        }
        
        // Set up camera preview layer
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.layer.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        
        // Set up video output
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(videoOutput)
        
        captureSession.startRunning()
    }
    
    func setupObjectDetection() {
        // Load the Core ML model
        guard let model = try? VNCoreMLModel(for: YOLOv3().model) else { return } // Replace 'YOLOv3' with your model class name
        
        // Create a Vision request for the model
        objectDetectionRequest = VNCoreMLRequest(model: model) { (request, error) in
            guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
            DispatchQueue.main.async {
                self.view.layer.sublayers?.removeSubrange(1...) // Clear previous boxes
                for result in results {
                    self.drawBoundingBox(for: result)
                }
            }
        }
        
        objectDetectionRequest.imageCropAndScaleOption = .scaleFill
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try? handler.perform([objectDetectionRequest])
    }
    
    func drawBoundingBox(for observation: VNRecognizedObjectObservation) {
        let boundingBox = observation.boundingBox
        let size = CGSize(width: boundingBox.width * view.bounds.width,
                          height: boundingBox.height * view.bounds.height)
        let origin = CGPoint(x: boundingBox.minX * view.bounds.width,
                             y: (1 - boundingBox.minY) * view.bounds.height - size.height)
        let boxLayer = CALayer()
        boxLayer.frame = CGRect(origin: origin, size: size)
        boxLayer.borderWidth = 2
        boxLayer.borderColor = UIColor.red.cgColor
        view.layer.addSublayer(boxLayer)
    }
}
