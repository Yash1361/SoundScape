import SwiftUI
import Vision
import ARKit

class ARModel: NSObject, ObservableObject, ARSessionDelegate {
    @Published var isDetecting = false
    @Published var detectedObjects: [DetectedObject] = []
    
    var arView: ARSCNView!
    private var detectionRequest: VNCoreMLRequest?
    private var visionModel: VNCoreMLModel?
    
    override init() {
        super.init()
        setupVision()
    }
    
    func setupVision() {
        guard let model = try? VNCoreMLModel(for: YOLOv3().model) else {
            fatalError("Failed to load Vision ML model")
        }
        visionModel = model
        detectionRequest = VNCoreMLRequest(model: model, completionHandler: handleDetections)
        detectionRequest?.imageCropAndScaleOption = .scaleFill
    }
    
    func startSession() {
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        arView.session.delegate = self
        arView.session.run(configuration)
    }
    
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isDetecting else { return }
        
        let pixelBuffer = frame.capturedImage
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
        do {
            try imageRequestHandler.perform([detectionRequest!])
        } catch {
            print("Failed to perform detection: \(error)")
        }
    }
    
    func handleDetections(request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            self.detectedObjects = []
            guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
            
            for observation in results {
                let bbox = observation.boundingBox
                let className = observation.labels[0].identifier
                let confidence = observation.confidence
                
                if let distance = self.calculateAccurateDistance(for: bbox) {
                    let object = DetectedObject(bbox: bbox,
                                                className: className,
                                                score: Float(confidence),
                                                distance: distance)
                    self.detectedObjects.append(object)
                }
            }
        }
    }
    
    func calculateAccurateDistance(for bbox: CGRect) -> Float? {
        guard let currentFrame = arView.session.currentFrame else { return nil }
        
        let viewportSize = arView.bounds.size
        let viewportPoint = CGPoint(x: bbox.midX * viewportSize.width,
                                    y: bbox.midY * viewportSize.height)
        
        guard let query = arView.raycastQuery(from: viewportPoint,
                                              allowing: .estimatedPlane,
                                              alignment: .any) else { return nil }
        
        guard let result = arView.session.raycast(query).first else { return nil }
        
        let worldPosition = result.worldTransform.columns.3
        let cameraPosition = currentFrame.camera.transform.columns.3
        
        let distance = simd_distance(worldPosition, cameraPosition)
        return distance
    }
}

struct DetectedObject: Identifiable {
    let id = UUID()
    let bbox: CGRect
    let className: String
    let score: Float
    let distance: Float
}

struct ContentView: View {
    @StateObject private var arModel = ARModel()
    
    var body: some View {
        ZStack {
            ARViewContainer(arModel: arModel)
                .edgesIgnoringSafeArea(.all)
            
            VStack {
                Spacer()
                Button(action: {
                    arModel.isDetecting.toggle()
                }) {
                    Text(arModel.isDetecting ? "Stop Detection" : "Start Detection")
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                }
                .padding(.bottom)
            }
            
            ForEach(arModel.detectedObjects) { object in
                ObjectBox(object: object)
            }
        }
    }
}

struct ARViewContainer: UIViewRepresentable {
    @ObservedObject var arModel: ARModel
    
    func makeUIView(context: Context) -> ARSCNView {
        let arView = ARSCNView(frame: .zero)
        arModel.arView = arView
        arModel.startSession()
        return arView
    }
    
    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

struct ObjectBox: View {
    let object: DetectedObject
    
    var body: some View {
        GeometryReader { geometry in
            let rect = CGRect(
                x: object.bbox.minX * geometry.size.width,
                y: object.bbox.minY * geometry.size.height,
                width: object.bbox.width * geometry.size.width,
                height: object.bbox.height * geometry.size.height
            )
            
            Rectangle()
                .stroke(Color.red, lineWidth: 2)
                .frame(width: rect.width, height: rect.height)
                .position(x: rect.midX, y: rect.midY)
            
            Text("\(object.className) - \(String(format: "%.2f", object.distance))m")
                .font(.caption)
                .foregroundColor(.white)
                .padding(4)
                .background(Color.black.opacity(0.7))
                .cornerRadius(4)
                .position(x: rect.minX, y: rect.minY - 10)
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
