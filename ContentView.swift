import SwiftUI
import Vision
import ARKit
import SceneKit
import AVFoundation

// MARK: - TargetAnchor Class
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

// MARK: - ARCombinedModel Class
class ARCombinedModel: NSObject, ObservableObject, ARSCNViewDelegate, ARSessionDelegate {
    // Published properties for UI updates
    @Published var isDetecting = false
    @Published var targetObject: String?
    @Published var availableObjects: [String] = []
    @Published var hasFoundTargetObject = false
    @Published var targetNodeVisible = false
    @Published var targetObjectScreenPosition: CGPoint?
    @Published var targetObjectScreenSize: CGSize?
    @Published var targetObjectDistance: Float?
    @Published var hasObstacle = false
    
    // ARKit properties
    var arView: ARSCNView!
    private var detectionRequest: VNCoreMLRequest?
    private var visionModel: VNCoreMLModel?
    private var targetNode: SCNNode?
    
    // Audio properties
    private var audioEngine: AVAudioEngine!
    private var guidePlayerNode: AVAudioPlayerNode!
    private var obstaclePlayerNode: AVAudioPlayerNode!
    private var pannerNode: AVAudioMixerNode!
    
    // Audio buffers
    private var guideBuffer: AVAudioPCMBuffer?
    private var obstacleBuffer: AVAudioPCMBuffer?
    
    // Obstacle detection properties
    private var obstacleDetectionTimer: Timer?
    private let safeZoneDistance: Float = 0.9
    private let directions: [SIMD3<Float>] = [
        SIMD3<Float>(0, 0, -1),
        SIMD3<Float>(-0.5, 0, -0.5),
        SIMD3<Float>(0.5, 0, -0.5),
        SIMD3<Float>(-1, 0, 0),
        SIMD3<Float>(1, 0, 0)
    ]
    
    override init() {
        super.init()
        setupVision()
        loadAvailableObjects()
        setupAudio()
    }
    
    // MARK: - Vision Setup
    func setupVision() {
        // Ensure you have a valid CoreML model named YOLOv3 in your project
        guard let model = try? VNCoreMLModel(for: YOLOv3().model) else {
            fatalError("Failed to load Vision ML model")
        }
        visionModel = model
        detectionRequest = VNCoreMLRequest(model: model, completionHandler: handleDetections)
        detectionRequest?.imageCropAndScaleOption = .scaleFill
        print("Vision setup completed")
    }
    
    func loadAvailableObjects() {
        availableObjects = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"].sorted()
        print("Available objects loaded")
    }
    
    // MARK: - Audio Setup
    func setupAudio() {
        audioEngine = AVAudioEngine()
        guidePlayerNode = AVAudioPlayerNode()
        obstaclePlayerNode = AVAudioPlayerNode()
        pannerNode = AVAudioMixerNode()
        
        // Attach nodes to the audio engine
        audioEngine.attach(guidePlayerNode)
        audioEngine.attach(obstaclePlayerNode)
        audioEngine.attach(pannerNode)
        
        // Connect guidePlayerNode -> pannerNode
        audioEngine.connect(guidePlayerNode, to: pannerNode, format: nil)
        // Connect pannerNode -> main mixer
        audioEngine.connect(pannerNode, to: audioEngine.mainMixerNode, format: nil)
        // Connect obstaclePlayerNode directly to main mixer
        audioEngine.connect(obstaclePlayerNode, to: audioEngine.mainMixerNode, format: nil)
        
        // Debug: Confirm connections
        print("AudioEngine connections established")
        
        // Load audio files
        guard let guideURL = Bundle.main.url(forResource: "continuousSound", withExtension: "wav"),
              let obstacleURL = Bundle.main.url(forResource: "static", withExtension: "wav") else {
            print("Error: Unable to find sound files 'continuousSound.wav' and 'static.wav'")
            return
        }
        
        do {
            // Load audio files
            let guideAudioFileTemp = try AVAudioFile(forReading: guideURL)
            let obstacleAudioFileTemp = try AVAudioFile(forReading: obstacleURL)
            print("Audio files loaded successfully")
            
            // Get main mixer format for buffer conversion
            let mainMixerFormat = audioEngine.mainMixerNode.outputFormat(forBus: 0)
            print("Main mixer format: \(mainMixerFormat.channelCount) channels, \(mainMixerFormat.sampleRate) Hz")
            
            // Convert guide audio buffer to match main mixer format
            let guideBufferTemp = try loadPCMBuffer(from: guideAudioFileTemp, to: mainMixerFormat)
            self.guideBuffer = guideBufferTemp
            print("Guide audio buffer converted to main mixer format")
            
            // Convert obstacle audio buffer to match main mixer format
            let obstacleBufferTemp = try loadPCMBuffer(from: obstacleAudioFileTemp, to: mainMixerFormat)
            self.obstacleBuffer = obstacleBufferTemp
            print("Obstacle audio buffer converted to main mixer format")
            
        } catch {
            print("Error loading audio files or creating buffers: \(error.localizedDescription)")
            return
        }
        
        do {
            // Configure audio session
            let audioSession = AVAudioSession.sharedInstance()
            try audioSession.setCategory(.playback, mode: .default, options: [])
            try audioSession.setActive(true)
            print("Audio session set up successfully")
        } catch {
            print("Error setting up audio session: \(error.localizedDescription)")
            return
        }
        
        do {
            // Start the audio engine
            try audioEngine.start()
            print("Audio engine started successfully")
        } catch {
            print("Error starting audio engine: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Helper Function to Convert Audio Files
    private func loadPCMBuffer(from file: AVAudioFile, to format: AVAudioFormat) throws -> AVAudioPCMBuffer {
        guard let converter = AVAudioConverter(from: file.processingFormat, to: format) else {
            print("Failed to create AVAudioConverter")
            throw NSError(domain: "AudioConversion", code: 0, userInfo: [NSLocalizedDescriptionKey: "Failed to create AVAudioConverter"])
        }
        
        // Calculate buffer capacity based on file length and sample rate
        let frameCapacity = AVAudioFrameCount(file.length)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCapacity)!
        
        var error: NSError?
        var currentFrame: AVAudioFramePosition = 0
        
        // Define the input block for AVAudioConverter
        let inputBlock: AVAudioConverterInputBlock = { inNumberOfPackets, outStatus in
            // Determine how many frames to read
            let framesToRead = min(AVAudioFrameCount(inNumberOfPackets), AVAudioFrameCount(file.length - currentFrame))
            if framesToRead == 0 {
                outStatus.pointee = .endOfStream
                return nil
            }
            
            // Create a temporary buffer to read data
            let tempBuffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: framesToRead)!
            do {
                try file.read(into: tempBuffer, frameCount: framesToRead)
                currentFrame += AVAudioFramePosition(framesToRead)
                outStatus.pointee = .haveData
                return tempBuffer
            } catch {
                print("Error reading audio file: \(error.localizedDescription)")
                outStatus.pointee = .endOfStream
                return nil
            }
        }
        
        // Perform the conversion
        try converter.convert(to: buffer, error: &error, withInputFrom: inputBlock)
        
        if let error = error {
            print("Error during audio conversion: \(error.localizedDescription)")
            throw error
        }
        
        return buffer
    }
    
    // MARK: - Audio Playback Methods
    func startGuideAudio() {
        guard let guideBuffer = guideBuffer else {
            print("Error: Guide buffer not loaded")
            return
        }
        
        // Ensure audio engine is running
        if !audioEngine.isRunning {
            do {
                try audioEngine.start()
                print("Audio engine started before playing guide audio")
            } catch {
                print("Error starting audio engine: \(error.localizedDescription)")
                return
            }
        }
        
        guidePlayerNode.stop()
        guidePlayerNode.scheduleBuffer(guideBuffer, at: nil, options: .loops, completionHandler: nil)
        guidePlayerNode.play()
        print("Guide audio started and looping")
    }
    
    func stopGuideAudio() {
        guidePlayerNode.stop()
        print("Guide audio stopped")
    }
    
    func startObstacleAudio() {
        guard let obstacleBuffer = obstacleBuffer else {
            print("Error: Obstacle buffer not loaded")
            return
        }
        
        // Ensure audio engine is running
        if !audioEngine.isRunning {
            do {
                try audioEngine.start()
                print("Audio engine started before playing obstacle audio")
            } catch {
                print("Error starting audio engine: \(error.localizedDescription)")
                return
            }
        }
        
        obstaclePlayerNode.stop()
        obstaclePlayerNode.scheduleBuffer(obstacleBuffer, at: nil, options: [], completionHandler: nil)
        obstaclePlayerNode.play()
        print("Obstacle audio started")
    }
    
    func stopObstacleAudio() {
        obstaclePlayerNode.stop()
        print("Obstacle audio stopped")
    }
    
    // MARK: - Audio Position Updates
    func updateGuideAudioPosition(targetPosition: simd_float3, listenerPosition: simd_float3, listenerForward: simd_float3) {
        let relativePosition = targetPosition - listenerPosition
        let distance = simd_length(relativePosition)
        
        let forward = simd_normalize(simd_make_float3(listenerForward.x, 0, listenerForward.z))
        let right = simd_cross(forward, SIMD3<Float>(0, 1, 0))
        let relativeDirection = simd_normalize(simd_make_float3(relativePosition.x, 0, relativePosition.z))
        
        let dotProduct = simd_dot(forward, relativeDirection)
        let angle = acos(dotProduct)
        
        let rightDotProduct = simd_dot(right, relativeDirection)
        let sign = rightDotProduct >= 0 ? 1.0 : -1.0
        
        let pan = Float(sign) * sin(angle)
        
        let baseVolume: Float
        if distance <= 0.5 {
            baseVolume = 1.0
        } else if distance >= 6.0 {
            baseVolume = 0.1
        } else {
            baseVolume = 1.0 - ((distance - 0.5) / 5.5) * 0.9
        }
        
        let angleAttenuation = cos(angle / 2)
        let volume = baseVolume * max(0.1, angleAttenuation)
        
        guidePlayerNode.volume = volume
        pannerNode.pan = pan
        
        print("Guide Audio - Angle: \(angle), Pan: \(pan), Volume: \(volume)")
    }
    
    func updateObstacleAudio(distance: Float, direction: SIMD3<Float>) {
        if distance < safeZoneDistance {
            if !hasObstacle {
                startObstacleAudio()
                hasObstacle = true
            }
            let volume = 1.0 - (distance / safeZoneDistance)
            let pan = direction.x
            obstaclePlayerNode.volume = volume
            obstaclePlayerNode.pan = pan
            print("Obstacle Audio - Distance: \(distance), Volume: \(volume), Pan: \(pan)")
        } else {
            if hasObstacle {
                stopObstacleAudio()
                hasObstacle = false
            }
        }
    }
    
    // MARK: - AR Session Management
    func startSession() {
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        configuration.frameSemantics.insert(.sceneDepth)
        arView.session.run(configuration)
        arView.delegate = self
        arView.session.delegate = self
        startObstacleDetection()
        print("AR session started")
    }
    
    func stopSession() {
        arView.session.pause()
        stopGuideAudio()
        stopObstacleAudio()
        obstacleDetectionTimer?.invalidate()
        obstacleDetectionTimer = nil
        print("AR session stopped")
    }
    
    func startObstacleDetection() {
        obstacleDetectionTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            self?.detectObstacles()
        }
        print("Obstacle detection started")
    }
    
    // MARK: - Obstacle Detection
    func detectObstacles() {
        guard let frame = arView.session.currentFrame else { return }
        
        let cameraTransform = frame.camera.transform
        let cameraPosition = simd_make_float3(cameraTransform.columns.3)
        let cameraForward = simd_make_float3(cameraTransform.columns.2)
        
        var closestDistance: Float = Float.greatestFiniteMagnitude
        var closestDirection: SIMD3<Float> = SIMD3<Float>(0, 0, -1)
        
        for direction in directions {
            // Convert direction to world space
            let worldDirection4 = cameraTransform * SIMD4<Float>(direction.x, direction.y, direction.z, 0)
            let worldDirection = simd_normalize(simd_make_float3(worldDirection4.x, worldDirection4.y, worldDirection4.z))
            
            // Perform raycast
            let query = ARRaycastQuery(origin: cameraPosition, direction: worldDirection, allowing: .estimatedPlane, alignment: .any)
            let results = arView.session.raycast(query)
            if let result = results.first {
                let distance = simd_distance(cameraPosition, simd_make_float3(result.worldTransform.columns.3))
                if distance < closestDistance {
                    closestDistance = distance
                    closestDirection = worldDirection
                }
            }
        }
        
        if closestDistance < safeZoneDistance {
            updateObstacleAudio(distance: closestDistance, direction: closestDirection)
        } else {
            updateObstacleAudio(distance: Float.greatestFiniteMagnitude, direction: SIMD3<Float>(0, 0, -1))
        }
    }
    
    // MARK: - ARSCNViewDelegate Methods
    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        guard let pointOfView = arView.pointOfView else { return }
        
        if let targetNode = targetNode {
            let targetPosition = targetNode.simdWorldPosition
            let cameraPosition = pointOfView.simdWorldPosition
            let cameraForward = pointOfView.simdWorldFront
            
            let distance = simd_distance(targetPosition, cameraPosition)
            
            let projectedPoint = arView.projectPoint(SCNVector3(targetPosition))
            let screenPoint = CGPoint(x: CGFloat(projectedPoint.x), y: CGFloat(projectedPoint.y))
            
            let size = CGFloat(0.2 / distance)
            let screenSize = CGSize(width: size * 100, height: size * 100)
            
            let isOnScreen = arView.bounds.contains(screenPoint)
            
            updateGuideAudioPosition(targetPosition: targetPosition, listenerPosition: cameraPosition, listenerForward: cameraForward)
            
            DispatchQueue.main.async {
                self.targetNodeVisible = isOnScreen
                self.targetObjectScreenPosition = screenPoint
                self.targetObjectScreenSize = screenSize
                self.targetObjectDistance = distance
            }
        }
    }
    
    // MARK: - ARSessionDelegate Methods
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if isDetecting {
            let pixelBuffer = frame.capturedImage
            let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .right, options: [:])
            do {
                try imageRequestHandler.perform([detectionRequest!])
            } catch {
                print("Failed to perform detection: \(error)")
            }
        }
    }
    
    func handleDetections(request: VNRequest, error: Error?) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self, let results = request.results as? [VNRecognizedObjectObservation] else { return }
            
            for observation in results {
                let className = observation.labels.first?.identifier ?? "Unknown"
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
                                    y: (1 - bbox.midY) * viewportSize.height) // y is flipped in Vision
        
        guard let query = arView.raycastQuery(from: viewportPoint,
                                              allowing: .estimatedPlane,
                                              alignment: .any) else {
            print("Failed to create raycast query from viewport point")
            return
        }
        
        let results = arView.session.raycast(query)
        if let result = results.first {
            let anchor = TargetAnchor(name: targetObject ?? "Unknown", transform: result.worldTransform)
            arView.session.add(anchor: anchor)
            
            let node = createTargetNode()
            node.simdTransform = result.worldTransform
            arView.scene.rootNode.addChildNode(node)
            targetNode = node
            
            hasFoundTargetObject = true
            isDetecting = false
            startGuideAudio()
            print("Found target object and started guide audio")
        } else {
            print("No raycast results found")
        }
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
        hasObstacle = false
        
        if let targetNode = targetNode {
            targetNode.removeFromParentNode()
        }
        targetNode = nil
        
        if let targetAnchor = arView.session.currentFrame?.anchors.compactMap({ $0 as? TargetAnchor }).first {
            arView.session.remove(anchor: targetAnchor)
        }
        
        stopGuideAudio()
        stopObstacleAudio()
        startSession()
        print("Session reset")
    }
}

// MARK: - ContentView
struct ContentView: View {
    @StateObject private var arModel = ARCombinedModel()
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
                        print(arModel.isDetecting ? "Detection started" : "Detection stopped")
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
                        print("Reset button pressed")
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
            
            // Highlight the target object on screen
            if let position = arModel.targetObjectScreenPosition,
               let size = arModel.targetObjectScreenSize,
               arModel.targetNodeVisible {
                Rectangle()
                    .stroke(Color.green, lineWidth: 2)
                    .frame(width: size.width, height: size.height)
                    .position(position)
            }
            
            // Display obstacle detected message
            if arModel.hasObstacle {
                VStack {
                    Spacer()
                    Text("Obstacle Detected!")
                        .font(.headline)
                        .padding()
                        .background(Color.red.opacity(0.7))
                        .foregroundColor(.white)
                        .cornerRadius(10)
                        .padding(.bottom, 100)
                }
            }
        }
        .sheet(isPresented: $showingObjectPicker) {
            ObjectPickerView(arModel: arModel)
        }
        .onAppear {
            arModel.startSession()
            print("ContentView appeared and AR session started")
        }
        .onDisappear {
            arModel.stopSession()
            print("ContentView disappeared and AR session stopped")
        }
    }
}

// MARK: - ARViewContainer
struct ARViewContainer: UIViewRepresentable {
    @ObservedObject var arModel: ARCombinedModel
    
    func makeUIView(context: Context) -> ARSCNView {
        let arView = ARSCNView(frame: .zero)
        arView.delegate = arModel
        arView.session.delegate = arModel
        arModel.arView = arView
        arModel.startSession()
        return arView
    }
    
    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}

// MARK: - ObjectPickerView
struct ObjectPickerView: View {
    @ObservedObject var arModel: ARCombinedModel
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        NavigationView {
            List(arModel.availableObjects, id: \.self) { object in
                Button(action: {
                    arModel.targetObject = object
                    presentationMode.wrappedValue.dismiss()
                    print("Selected object: \(object)")
                }) {
                    Text(object)
                }
            }
            .navigationTitle("Select Target Object")
        }
    }
}

// MARK: - Extensions
extension simd_float4x4 {
    func rotate(vector: SIMD3<Float>) -> SIMD3<Float> {
        let rotatedVector = SIMD3<Float>(
            x: columns.0.x * vector.x + columns.1.x * vector.y + columns.2.x * vector.z,
            y: columns.0.y * vector.x + columns.1.y * vector.y + columns.2.y * vector.z,
            z: columns.0.z * vector.x + columns.1.z * vector.y + columns.2.z * vector.z
        )
        return rotatedVector
    }
}

extension simd_float4x4 {
    var orientation: AVAudio3DAngularOrientation {
        let forward = -columns.2.xyz
        let up = columns.1.xyz
        
        let yaw = atan2f(forward.x, forward.z) * (180.0 / .pi)
        let pitch = asinf(forward.y) * (180.0 / .pi)
        let roll = atan2f(up.x, up.y) * (180.0 / .pi)
        
        return AVAudio3DAngularOrientation(yaw: yaw, pitch: pitch, roll: roll)
    }
}

extension simd_float4 {
    var xyz: SIMD3<Float> {
        return SIMD3<Float>(x, y, z)
    }
}

extension SIMD3 where Scalar == Float {
    var normalized: SIMD3<Float> {
        return self / length(self)
    }
}

// MARK: - Preview
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
