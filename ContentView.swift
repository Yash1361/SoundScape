import SwiftUI
import RealityKit
import ARKit
import AVFoundation

struct ContentView: View {
    @StateObject private var arManager = ARManager()
    @State private var isScanningActive = false

    var body: some View {
        ZStack {
            ARViewContainer(arManager: arManager).edgesIgnoringSafeArea(.all)
            VStack {
                Spacer()
                Button(isScanningActive ? "Stop Scanning" : "Start Scanning") {
                    if isScanningActive {
                        arManager.stopSession()
                    } else {
                        arManager.startSession()
                    }
                    isScanningActive.toggle()
                }
                .padding()
                .background(isScanningActive ? Color.red : Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
                .padding(.bottom, 50)
            }
        }
    }
}

struct ARViewContainer: UIViewRepresentable {
    var arManager: ARManager

    func makeUIView(context: Context) -> ARView {
        arManager.arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {}
}

class ARManager: NSObject, ObservableObject {
    @Published var arView: ARView
    var detectedPlanes: [ARPlaneAnchor: AnchorEntity] = [:]
    var updateTimer: Timer?

    // Audio engine components for spatial audio
    var audioEngine: AVAudioEngine?
    var environmentNode: AVAudioEnvironmentNode?
    var audioPlayerNode: AVAudioPlayerNode?
    var audioFile: AVAudioFile?
    let safeZoneDistance: Float = 0.9 // Define the safe zone threshold distance

    override init() {
        arView = ARView(frame: .zero)
        super.init()
        setupAudioSession()
        setupSpatialAudio()
    }

    // ... (keep the existing audio setup methods)

    func setupAudioSession() {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playback, mode: .default, options: [])
            try audioSession.setActive(true)
            print("Audio session set up successfully")
        } catch {
            print("Failed to set up audio session: \(error.localizedDescription)")
        }
    }

    func setupSpatialAudio() {
        audioEngine = AVAudioEngine()
        environmentNode = AVAudioEnvironmentNode()
        audioPlayerNode = AVAudioPlayerNode()

        guard let audioEngine = audioEngine,
              let environmentNode = environmentNode,
              let audioPlayerNode = audioPlayerNode,
              let audioFileURL = Bundle.main.url(forResource: "continuousSound", withExtension: "wav") else {
            print("Failed to initialize audio components")
            return
        }

        do {
            audioFile = try AVAudioFile(forReading: audioFileURL)
            audioEngine.attach(environmentNode)
            audioEngine.attach(audioPlayerNode)

            // Set the rendering algorithm for spatial audio
            audioPlayerNode.renderingAlgorithm = .HRTF

            audioEngine.connect(audioPlayerNode, to: environmentNode, format: audioFile?.processingFormat)
            audioEngine.connect(environmentNode, to: audioEngine.mainMixerNode, format: nil)

            // Start the audio engine
            try audioEngine.start()
            print("Audio Engine started")

            // Schedule the audio buffer but do not start playing yet
            scheduleAudioBuffer()
            print("Spatial audio setup complete")
        } catch {
            print("Error setting up spatial audio: \(error)")
        }
    }

    func scheduleAudioBuffer() {
        guard let audioFile = audioFile else { return }
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            print("Failed to create PCM buffer")
            return
        }
        do {
            try audioFile.read(into: buffer)
        } catch {
            print("Error reading audio file into buffer: \(error)")
            return
        }

        audioPlayerNode?.scheduleBuffer(buffer, at: nil, options: [.loops, .interrupts], completionHandler: nil)
        audioPlayerNode?.play()
    }

    func updateAudioPosition(distance: Float, direction: SIMD3<Float>) {
        guard let environmentNode = environmentNode,
              let audioPlayerNode = audioPlayerNode,
              let frame = arView.session.currentFrame else { return }

        let safeZoneDistance: Float = 0.9 // Adjust as needed

        // Calculate the volume based on proximity
        let volume = max(0, min(1, 1 - (distance / safeZoneDistance)))
        audioPlayerNode.volume = volume

        // Update the audio position relative to the listener
        let audioPosition = AVAudio3DPoint(x: direction.x * distance, y: direction.y * distance, z: direction.z * distance)
        audioPlayerNode.position = audioPosition

        // Update the listener's position and orientation
        let cameraTransform = frame.camera.transform
        let cameraPosition = cameraTransform.columns.3.xyz
        environmentNode.listenerPosition = AVAudio3DPoint(x: cameraPosition.x, y: cameraPosition.y, z: cameraPosition.z)

        // Extract orientation from camera transform
        let cameraOrientation = cameraTransform.orientation
        environmentNode.listenerAngularOrientation = cameraOrientation

        print("Updated audio position to \(audioPosition), volume: \(volume)")
    }

    func startSession() {
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]

        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh
            print("LiDAR scanning enabled")
        } else {
            print("LiDAR not available on this device")
        }

        if type(of: config).supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics = .sceneDepth
        }

        arView.session.delegate = self
        arView.session.run(config)

        startUpdatingDistance()
    }

    func stopSession() {
        arView.session.pause()
        updateTimer?.invalidate()
        audioEngine?.stop()
    }

    func startUpdatingDistance() {
        if updateTimer == nil {
            updateTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
                self?.updateDistanceToNearestObject()
            }
        }
    }

    let directions: [(name: String, vector: SIMD3<Float>)] = [
        ("forward", SIMD3<Float>(0, 0, -1)),
        ("down-forward", SIMD3<Float>(0, -0.5, -0.5).normalized),
        ("left", SIMD3<Float>(-1, 0, 0)),
        ("right", SIMD3<Float>(1, 0, 0)),
        ("down-left", SIMD3<Float>(-0.5, -0.5, -0.5).normalized),
        ("down-right", SIMD3<Float>(0.5, -0.5, -0.5).normalized),
        ("slightly-down", SIMD3<Float>(0, -0.2, -0.8).normalized),
        ("far-left", SIMD3<Float>(-0.7071, 0, -0.7071)),
        ("far-right", SIMD3<Float>(0.7071, 0, -0.7071)),
        // New directions for low obstacles
        ("very-down-forward", SIMD3<Float>(0, -0.7, -0.3).normalized),
        ("very-down-left", SIMD3<Float>(-0.3, -0.7, -0.3).normalized),
        ("very-down-right", SIMD3<Float>(0.3, -0.7, -0.3).normalized),
        ("floor-check", SIMD3<Float>(0, -1, 0))
    ]
    func updateDistanceToNearestObject() {
        guard let frame = arView.session.currentFrame,
              let depthMap = frame.sceneDepth?.depthMap else { return }

        let cameraTransform = frame.camera.transform
        let cameraPosition = cameraTransform.columns.3.xyz

        // Get the current interface orientation
        let interfaceOrientation = getCurrentInterfaceOrientation()

        // Get the view matrix for the current orientation
        let viewMatrix = frame.camera.viewMatrix(for: interfaceOrientation)

        var closestObstacle: (direction: String, distance: Float, worldDirection: SIMD3<Float>)?

        for (name, dir) in directions {
            let worldDirection = simd_normalize(cameraTransform.rotate(vector: dir))

            if let depth = getDepth(for: worldDirection, in: depthMap, viewMatrix: viewMatrix) {
                let distance = depth * length(worldDirection)

                if closestObstacle == nil || distance < closestObstacle!.distance {
                    closestObstacle = (direction: name, distance: distance, worldDirection: worldDirection)
                }
            }
        }

        if let obstacle = closestObstacle {
            print("Detected obstacle \(obstacle.direction) at distance \(obstacle.distance)")
            updateAudioFeedback(distance: obstacle.distance, direction: obstacle.worldDirection)
        } else {
            updateAudioFeedback(distance: Float.greatestFiniteMagnitude, direction: SIMD3<Float>(0, 0, -1))
        }
    }

    // Helper function to get the current interface orientation
    private func getCurrentInterfaceOrientation() -> UIInterfaceOrientation {
        guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene else {
            return .portrait
        }
        return windowScene.interfaceOrientation
    }

    func updateAudioFeedback(distance: Float, direction: SIMD3<Float>) {
        guard let audioPlayerNode = audioPlayerNode else { return }

        let safeZoneDistance: Float = 0.9  // 0.9 meters
        let minimumAudibleVolume: Float = 0.1  // Adjust this value as needed

        if distance > safeZoneDistance {
            audioPlayerNode.volume = 0
            print("Debug: Object beyond safe zone. Audio muted.")
            return
        }

        var volume = 1 - (distance / safeZoneDistance)

        // Apply minimum volume threshold
        if volume > 0 && volume < minimumAudibleVolume {
            volume = 0
        }

        let pan = direction.x

        audioPlayerNode.volume = volume
        audioPlayerNode.pan = pan

        let audioPosition = AVAudio3DPoint(x: direction.x * distance, y: direction.y * distance, z: direction.z * distance)
        audioPlayerNode.position = audioPosition

        print("Debug: Updated audio - distance: \(distance), volume: \(volume), pan: \(pan)")
    }

    func getDepth(for direction: SIMD3<Float>, in depthMap: CVPixelBuffer, viewMatrix: simd_float4x4) -> Float? {
        let textureSize = CGSize(width: CVPixelBufferGetWidth(depthMap), height: CVPixelBufferGetHeight(depthMap))
        let projectedPoint = simd_float4(direction.x, direction.y, direction.z, 1) * viewMatrix
        let normalizedPoint = simd_float2(projectedPoint.x, projectedPoint.y) / projectedPoint.z
        let texturePoint = simd_float2((normalizedPoint.x + 1) / 2, (1 - normalizedPoint.y) / 2)

        let x = Int(texturePoint.x * Float(textureSize.width))
        let y = Int(texturePoint.y * Float(textureSize.height))

        guard x >= 0 && x < Int(textureSize.width) && y >= 0 && y < Int(textureSize.height) else {
            print("Point out of bounds")
            return nil
        }

        var depth: Float = 0

        // Lock the base address of the pixel buffer
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer {
            // Unlock the base address when we're done
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
        }

        // Check if we can access the pixel data
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthMap) else {
            print("Unable to access depth map data")
            return nil
        }

        // Check the pixel format to ensure it's what we expect (32-bit float)
        let pixelFormat = CVPixelBufferGetPixelFormatType(depthMap)
        guard pixelFormat == kCVPixelFormatType_DepthFloat32 else {
            print("Unexpected pixel format")
            return nil
        }

        // Get the bytes per row to calculate the correct offset
        let bytesPerRow = CVPixelBufferGetBytesPerRow(depthMap)

        // Calculate the offset for the pixel we want
        let offset = y * bytesPerRow + x * MemoryLayout<Float>.size

        // Read the depth value
        memcpy(&depth, baseAddress.advanced(by: offset), MemoryLayout<Float>.size)

        return depth
    }

    func addPlaneVisualization(to anchor: ARPlaneAnchor) {
            let planeAnchor = AnchorEntity(anchor: anchor)

            let planeMesh = MeshResource.generatePlane(width: anchor.planeExtent.width, depth: anchor.planeExtent.height)
            let planeMaterial = SimpleMaterial(color: anchor.alignment == .horizontal ? .blue.withAlphaComponent(0.5) : .green.withAlphaComponent(0.5), isMetallic: false)
            let planeEntity = ModelEntity(mesh: planeMesh, materials: [planeMaterial])

            planeEntity.position = SIMD3(anchor.center.x, 0, anchor.center.z)
            planeAnchor.addChild(planeEntity)

            arView.scene.addAnchor(planeAnchor)
            detectedPlanes[anchor] = planeAnchor
        }
    func setupAppLifecycleNotifications() {
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(pauseSession),
                                               name: UIApplication.didEnterBackgroundNotification,
                                               object: nil)
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(resumeSession),
                                               name: UIApplication.willEnterForegroundNotification,
                                               object: nil)
    }

    @objc func pauseSession() {
        stopSession() // Pause AR session and stop timers
    }

    @objc func resumeSession() {
        startSession() // Resume AR session when app comes to the foreground
    }

}

extension ARManager: ARSessionDelegate {
    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
        for anchor in anchors {
            if let planeAnchor = anchor as? ARPlaneAnchor {
                print("Detected \(planeAnchor.alignment == .horizontal ? "horizontal" : "vertical") plane")
                addPlaneVisualization(to: planeAnchor)
            }
        }
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        for anchor in anchors {
            if let planeAnchor = anchor as? ARPlaneAnchor, let planeEntity = detectedPlanes[planeAnchor] {
                updatePlaneVisualization(planeAnchor: planeAnchor, planeEntity: planeEntity)
            }
        }
    }

    func updatePlaneVisualization(planeAnchor: ARPlaneAnchor, planeEntity: AnchorEntity) {
            if let planeModelEntity = planeEntity.children.first as? ModelEntity {
                planeModelEntity.model?.mesh = .generatePlane(width: planeAnchor.planeExtent.width, depth: planeAnchor.planeExtent.height)
                planeModelEntity.position = SIMD3(planeAnchor.center.x, 0, planeAnchor.center.z)
            }
        }
}

extension simd_float4x4 {
    func rotate(vector: SIMD3<Float>) -> SIMD3<Float> {
        // Multiply the vector by the rotation matrix (upper-left 3x3 of the transform)
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
        // Extract forward vector (negative z-axis)
        let forward = -columns.2.xyz
        // Extract up vector (y-axis)
        let up = columns.1.xyz

        // Calculate yaw (rotation around y-axis)
        let yaw = atan2f(forward.x, forward.z) * (180.0 / .pi)

        // Calculate pitch (rotation around x-axis)
        let pitch = asinf(forward.y) * (180.0 / .pi)

        // Calculate roll (rotation around z-axis)
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

#Preview {
    ContentView()
}
