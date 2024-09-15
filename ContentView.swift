import SwiftUI
import Vision
import ARKit
import SceneKit

class TargetAnchor: ARAnchor {
    var objectName: String
    
    override init(name: String, transform: simd_float4x4) {
        self.objectName = name
        super.init(name: name, transform: transform)
    }
    
    required init(anchor: ARAnchor) {
        self.objectName = anchor.name ?? "Unknown"
        super.init(anchor: anchor)
    }
    
    required init?(coder aDecoder: NSCoder) {
        self.objectName = aDecoder.decodeObject(forKey: "objectName") as? String ?? "Unknown"
        super.init(coder: aDecoder)
    }
    
    override func encode(with aCoder: NSCoder) {
        super.encode(with: aCoder)
        aCoder.encode(objectName, forKey: "objectName")
    }
    
    override class var supportsSecureCoding: Bool {
        return true
    }
}

class ARModel: NSObject, ObservableObject, ARSCNViewDelegate {
    @Published var isDetecting = false
    @Published var targetObject: String?
    @Published var availableObjects: [String] = []
    @Published var hasFoundTargetObject = false
    @Published var targetNodeVisible = false
    @Published var targetObjectScreenPosition: CGPoint?
    @Published var targetObjectScreenSize: CGSize?
    @Published var targetObjectDistance: Float?
    
    var arView: ARSCNView!
    private var detectionRequest: VNCoreMLRequest?
    private var visionModel: VNCoreMLModel?
    private var targetNode: SCNNode?
    
    override init() {
        super.init()
        setupVision()
        loadAvailableObjects()
    }
    
    func setupVision() {
        guard let model = try? VNCoreMLModel(for: YOLOv3().model) else {
            fatalError("Failed to load Vision ML model")
        }
        visionModel = model
        detectionRequest = VNCoreMLRequest(model: model, completionHandler: handleDetections)
        detectionRequest?.imageCropAndScaleOption = .scaleFill
    }
    
    func loadAvailableObjects() {
        availableObjects = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"].sorted()
    }
    
    func startSession() {
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        configuration.frameSemantics.insert(.sceneDepth)
        arView.session.run(configuration)
        arView.delegate = self
        arView.session.delegate = self
    }
    
    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        guard let pointOfView = arView.pointOfView else { return }
        
        if let targetNode = targetNode {
            let targetPosition = targetNode.simdWorldPosition
            let cameraPosition = pointOfView.simdWorldPosition
            
            let distance = simd_distance(targetPosition, cameraPosition)
            
            let projectedPoint = arView.projectPoint(SCNVector3(targetPosition))
            let screenPoint = CGPoint(x: CGFloat(projectedPoint.x), y: CGFloat(projectedPoint.y))
            
            let size = CGFloat(0.2 / distance)
            let screenSize = CGSize(width: size * 100, height: size * 100)
            
            let isOnScreen = arView.bounds.contains(screenPoint)
            
            DispatchQueue.main.async {
                self.targetNodeVisible = isOnScreen
                self.targetObjectScreenPosition = screenPoint
                self.targetObjectScreenSize = screenSize
            }
        }
    }
    
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if isDetecting {
            let pixelBuffer = frame.capturedImage
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
            do {
                try imageRequestHandler.perform([detectionRequest!])
            } catch {
                print("Failed to perform detection: \(error)")
            }
        } else if hasFoundTargetObject {
            updateTargetDistance(frame: frame)
        }
    }
    
    func updateTargetDistance(frame: ARFrame) {
        guard let targetNode = targetNode,
              let sceneDepth = frame.sceneDepth,
              let devicePosition = arView.pointOfView?.simdWorldPosition else { return }
        
        let targetPosition = targetNode.simdWorldPosition
        
        let projectedPoint = arView.projectPoint(SCNVector3(targetPosition))
        
        let depthWidth = CGFloat(CVPixelBufferGetWidth(sceneDepth.depthMap))
        let depthHeight = CGFloat(CVPixelBufferGetHeight(sceneDepth.depthMap))
        
        let scaleX = depthWidth / frame.camera.imageResolution.width
        let scaleY = depthHeight / frame.camera.imageResolution.height
        
        let depthX = Int(CGFloat(projectedPoint.x) * scaleX)
        let depthY = Int(CGFloat(projectedPoint.y) * scaleY)
        
        guard depthX >= 0 && depthX < Int(depthWidth) &&
              depthY >= 0 && depthY < Int(depthHeight) else { return }
        
        CVPixelBufferLockBaseAddress(sceneDepth.depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(sceneDepth.depthMap, .readOnly) }
        
        guard let depthPointer = CVPixelBufferGetBaseAddress(sceneDepth.depthMap)?.assumingMemoryBound(to: Float32.self) else {
            return
        }
        
        let depthValue = depthPointer[depthY * Int(depthWidth) + depthX]
        
        let depthInMeters = depthValue
        
        let distance = simd_distance(devicePosition, targetPosition)
        
        DispatchQueue.main.async {
            self.targetObjectDistance = distance
        }
    }
    
    func handleDetections(request: VNRequest, error: Error?) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self, let results = request.results as? [VNRecognizedObjectObservation] else { return }
            
            for observation in results {
                let className = observation.labels[0].identifier
                if className == self.targetObject {
                    self.foundTargetObject(observation: observation)
                    return
                }
            }
        }
    }
    
    func foundTargetObject(observation: VNRecognizedObjectObservation) {
        guard let currentFrame = arView.session.currentFrame else { return }
        
        let bbox = observation.boundingBox
        let viewportSize = arView.bounds.size
        let viewportPoint = CGPoint(x: bbox.midX * viewportSize.width,
                                    y: bbox.midY * viewportSize.height)
        
        guard let query = arView.raycastQuery(from: viewportPoint,
                                              allowing: .estimatedPlane,
                                              alignment: .any) else { return }
        
        guard let result = arView.session.raycast(query).first else { return }
        
        let anchor = TargetAnchor(name: targetObject ?? "Unknown", transform: result.worldTransform)
        arView.session.add(anchor: anchor)
        
        let node = createTargetNode()
        node.simdTransform = result.worldTransform
        arView.scene.rootNode.addChildNode(node)
        targetNode = node
        
        hasFoundTargetObject = true
        isDetecting = false
    }
    
    func createTargetNode() -> SCNNode {
        let boxGeometry = SCNBox(width: 0.1, height: 0.1, length: 0.1, chamferRadius: 0)
        let material = SCNMaterial()
        material.diffuse.contents = UIColor.green.withAlphaComponent(0.7)
        boxGeometry.materials = [material]
        
        let node = SCNNode(geometry: boxGeometry)
        node.name = "TargetNode"
        
        return node
    }
    
    func reset() {
        hasFoundTargetObject = false
        targetObject = nil
        isDetecting = false
        targetNodeVisible = false
        targetObjectScreenPosition = nil
        targetObjectScreenSize = nil
        targetObjectDistance = nil
        
        if let targetNode = targetNode {
            targetNode.removeFromParentNode()
        }
        targetNode = nil
        
        if let targetAnchor = arView.session.currentFrame?.anchors.compactMap({ $0 as? TargetAnchor }).first {
            arView.session.remove(anchor: targetAnchor)
        }
        
        startSession()
    }
}

extension ARModel: ARSessionDelegate {
    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for anchor in anchors {
            if let targetAnchor = anchor as? TargetAnchor {
                let node = createTargetNode()
                node.simdTransform = targetAnchor.transform
                arView.scene.rootNode.addChildNode(node)
                targetNode = node
            }
        }
    }
}

struct ContentView: View {
    @StateObject private var arModel = ARModel()
    @State private var showingObjectPicker = false
    
    var body: some View {
        ZStack {
            ARViewContainer(arModel: arModel)
                .edgesIgnoringSafeArea(.all)
            
            VStack {
                if !arModel.hasFoundTargetObject {
                    if !arModel.isDetecting {
                        Button(action: {
                            showingObjectPicker = true
                        }) {
                            Text("Find Item")
                                .padding()
                                .background(Color.blue)
                                .foregroundColor(.white)
                                .cornerRadius(10)
                        }
                        .padding(.top)
                    }
                    
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
                } else {
                    Text("Target object found!")
                        .font(.headline)
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                        .padding(.top)
                    
                    if arModel.targetNodeVisible {
                        Text("Target object visible")
                            .font(.subheadline)
                            .padding()
                            .background(Color.yellow)
                            .foregroundColor(.black)
                            .cornerRadius(10)
                            .padding(.top)
                    }
                    
                    if let distance = arModel.targetObjectDistance {
                        Text(String(format: "Distance: %.2f meters", distance))
                            .font(.subheadline)
                            .padding()
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                            .padding(.top)
                    }
                    
                    Spacer()
                    
                    Button(action: {
                        arModel.reset()
                    }) {
                        Text("Reset")
                            .padding()
                            .background(Color.red)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    .padding(.bottom)
                }
            }
            
            if let position = arModel.targetObjectScreenPosition,
               let size = arModel.targetObjectScreenSize,
               arModel.targetNodeVisible {
                Rectangle()
                    .stroke(Color.green, lineWidth: 2)
                    .frame(width: size.width, height: size.height)
                    .position(position)
            }
        }
        .sheet(isPresented: $showingObjectPicker) {
            ObjectPickerView(arModel: arModel)
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

struct ObjectPickerView: View {
    @ObservedObject var arModel: ARModel
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        NavigationView {
            List(arModel.availableObjects, id: \.self) { object in
                Button(action: {
                    arModel.targetObject = object
                    presentationMode.wrappedValue.dismiss()
                }) {
                    Text(object)
                }
            }
            .navigationTitle("Select Target Object")
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
