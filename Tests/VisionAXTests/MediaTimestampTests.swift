//
//  MediaTimestampTests.swift
//  VisionAXTests
//
//  WHAT: Reading a player's clock out of text, and refusing to read anyone else's.
//  OUT:  MediaTimestamp
//

import CoreGraphics
import Foundation
import Testing
@testable import VisionAX

@Suite struct MediaTimestampTests {

    private func run(_ text: String, x: CGFloat = 40, y: CGFloat = 660) -> TextRun {
        TextRun(string: text, frame: CGRect(x: x, y: y, width: 90, height: 16), confidence: 0.9)
    }

    @Test func aPairReadsAsPositionThenLength() {
        let clock = MediaTimestamp.parse([run("1:23 / 4:56")], within: nil)
        #expect(clock.elapsed == 83)
        #expect(clock.duration == 296)
    }

    @Test func hoursAreUnderstood() {
        let clock = MediaTimestamp.parse([run("1:02:03 / 2:00:00")], within: nil)
        #expect(clock.elapsed == 3723)
        #expect(clock.duration == 7200)
    }

    /// Two separate runs beside each other are the same clock — plenty of players draw
    /// the position and the length as different elements.
    @Test func twoRunsMakeOneClock() {
        let clock = MediaTimestamp.parse(
            [run("4:56", x: 120), run("0:41", x: 40)], within: nil)
        #expect(clock.elapsed == 41, "the LEFTMOST timecode is the position")
        #expect(clock.duration == 296)
    }

    /// A LONE TIME IS THE POSITION. Guessing it was the length would make every seek
    /// look like it landed at the wrong place.
    @Test func aLoneTimeIsElapsed() {
        let clock = MediaTimestamp.parse([run("0:41")], within: nil)
        #expect(clock.elapsed == 41)
        #expect(clock.duration == nil)
    }

    /// Safari's native controls count DOWN, and a negative figure names what is left.
    @Test func aCountdownIsRemainingNotElapsed() {
        let clock = MediaTimestamp.parse([run("-3:33")], within: nil)
        #expect(clock.remaining == 213)
        #expect(clock.elapsed == nil)
    }

    @Test func aCountdownBesideATotalYieldsThePosition() {
        let clock = MediaTimestamp.parse([run("-3:33 5:00")], within: nil)
        #expect(clock.remaining == 213)
        #expect(clock.duration == 300)
        #expect(clock.elapsed == 87)
    }

    /// ONLY THE BAR'S TEXT IS THE PLAYER'S CLOCK. A comment quoting a timecode is on
    /// the page too, and reading it would report a stranger's number as this video's.
    @Test func textOutsideTheBarIsNotTheClock() {
        let bar = CGRect(x: 0, y: 640, width: 1280, height: 80)
        let comment = run("skip to 2:15 for the good part", x: 300, y: 200)
        #expect(MediaTimestamp.parse([comment], within: bar).elapsed == nil)
        #expect(MediaTimestamp.parse([comment], within: nil).elapsed == 135,
                "with no bar to scope to, the caller has taken responsibility")
    }

    @Test func textWithNoTimecodeIsNoClock() {
        let clock = MediaTimestamp.parse([run("Subscribe"), run("1.2M views")], within: nil)
        #expect(clock.elapsed == nil)
        #expect(clock.duration == nil)
    }

    /// SIXTY SECONDS IS NOT A TIMECODE. A version number or a score would otherwise
    /// parse, and the resulting "duration" would be nonsense nobody could trace.
    @Test func anImpossibleSecondsFieldIsRefused() {
        #expect(MediaTimestamp.seconds(in: "1:99") == nil)
    }

    /// A duration below the position is a misread, and dropping it beats publishing a
    /// video that ends before it starts.
    @Test func aBackwardsPairDropsTheDuration() {
        let clock = MediaTimestamp.parse([run("4:56 / 1:23")], within: nil)
        #expect(clock.elapsed == 296)
        #expect(clock.duration == nil)
    }
}
