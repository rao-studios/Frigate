import Testing
@testable import Frigate

@Suite("Frigate")
struct FrigateTests {

    @Test("FrigateEmbedder initializes without error")
    func embedderInit() {
        let _ = FrigateEmbedder()
    }

    @Test("FrigateLLM initializes without error")
    func llmInit() {
        let _ = FrigateLLM()
    }
}
