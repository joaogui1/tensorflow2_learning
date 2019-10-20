import XCTest
@testable import irisTutorial

final class irisTutorialTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(irisTutorial().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
