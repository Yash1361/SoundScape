import SwiftUI
import Speech
import AVFoundation

enum FindOption: String, CaseIterable {
    case bottle = "bottle"
    case backpack = "backpack"
    case cellphone = "cellphone"
}

class SpeechRecognizer: ObservableObject {
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))!
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let audioEngine = AVAudioEngine()
    private var silenceTimer: Timer?
    private let synthesizer = AVSpeechSynthesizer()
    
    @Published var isListening = false
    @Published var recognizedText = ""
    @Published var selectedOption: FindOption?
    @Published var isFinished = false
    
    func startListening() {
        guard !isFinished else { return }
        
        SFSpeechRecognizer.requestAuthorization { authStatus in
            DispatchQueue.main.async {
                if authStatus == .authorized {
                    do {
                        try self.startRecording()
                        self.isListening = true
                        self.speak("What would you like to find today?")
                    } catch {
                        print("Failed to start recording: \(error)")
                    }
                } else {
                    print("Speech recognition not authorized")
                }
            }
        }
    }
    
    func stopListening() {
        audioEngine.stop()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        self.isListening = false
        silenceTimer?.invalidate()
    }
    
    private func startRecording() throws {
        recognitionTask?.cancel()
        self.recognitionTask = nil
        
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.playAndRecord, mode: .default, options: .defaultToSpeaker)
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        
        let inputNode = audioEngine.inputNode
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else { fatalError("Unable to create a SFSpeechAudioBufferRecognitionRequest object") }
        recognitionRequest.shouldReportPartialResults = true
        
        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { result, error in
            var isFinal = false
            if let result = result {
                self.recognizedText = result.bestTranscription.formattedString
                isFinal = result.isFinal
                self.resetSilenceTimer()
            }
            
            if error != nil || isFinal {
                self.audioEngine.stop()
                inputNode.removeTap(onBus: 0)
                self.recognitionRequest = nil
                self.recognitionTask = nil
                self.interpretCommand()
            }
        }
        
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { (buffer: AVAudioPCMBuffer, when: AVAudioTime) in
            self.recognitionRequest?.append(buffer)
        }
        
        audioEngine.prepare()
        try audioEngine.start()
        
        resetSilenceTimer()
    }
    
    private func resetSilenceTimer() {
        silenceTimer?.invalidate()
        silenceTimer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: false) { _ in
            self.stopListening()
            self.interpretCommand()
        }
    }
    
    private func interpretCommand() {
        let lowercasedText = recognizedText.lowercased()
        
        if lowercasedText.contains("bottle") || lowercasedText.contains("water") {
            selectedOption = .bottle
        } else if lowercasedText.contains("backpack") || lowercasedText.contains("bag") {
            selectedOption = .backpack
        } else if lowercasedText.contains("cellphone") || lowercasedText.contains("phone") || lowercasedText.contains("mobile") {
            selectedOption = .cellphone
        }
        
        if let option = selectedOption {
            speak("Got it. Searching for \(option.rawValue) now.")
            isFinished = true
        } else {
            speak("Sorry, I didn't catch that. Please try again.")
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                self.recognizedText = ""
                self.startListening()
            }
        }
    }
    
    func speak(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        synthesizer.speak(utterance)
    }
    
    func reset() {
        isFinished = false
        selectedOption = nil
        recognizedText = ""
        startListening()
    }
}

struct ContentView: View {
    @StateObject private var speechRecognizer = SpeechRecognizer()
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Voice Command Object Finder")
                .font(.largeTitle)
            
            ForEach(FindOption.allCases, id: \.self) { option in
                Text(option.rawValue.capitalized)
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(speechRecognizer.selectedOption == option ? Color.blue : Color.gray)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            
            Button(action: {
                speechRecognizer.reset()
            }) {
                Text("Reset")
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
            
            Text("Status: \(speechRecognizer.isListening ? "Listening" : (speechRecognizer.isFinished ? "Finished" : "Idle"))")
                .padding()
            
            Text("Recognized: \(speechRecognizer.recognizedText)")
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.gray.opacity(0.2))
                .cornerRadius(10)
        }
        .padding()
        .onAppear {
            speechRecognizer.startListening()
        }
    }
}
