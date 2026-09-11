//
//  MediaControlTests.swift
//  VisionAXTests
//
//  WHAT: The media lane, against a drawn player whose transport is known.
//  OUT:  SyntheticPlayerSupport
//  PIN:  Every assertion here is about the PIPELINE — did the band scan find a 5px
//        track, did the fraction match where it was drawn, did the slots land on the
//        right buttons, did the witnesses say what they saw. Glyph fidelity against
//        real artwork is the live probe's job.
//

import CoreGraphics
import Foundation
import Testing
@testable import VisionAX

@Suite struct MediaControlTests {

    // MARK: - The track

    /// THE TRACK IS FOUND AND IT IS WHERE IT WAS DRAWN. A 5px bar is under the region
    /// detector's 8px floor by design — the band scan is a separate instrument, and
    /// this is the assertion that says so.
    @Test func theProgressTrackIsFoundAndMeasured() throws {
        let engine = try VisionEngine()
        let image = SyntheticPlayer.image(fraction: 0.35)
        let reading = try engine.readMediaControls(in: image)

        #expect(reading.controlsVisible)
        let progress = try #require(reading.progress)
        #expect(abs(progress.fraction - 0.35) < 0.02)
        #expect(abs(progress.frame.minY - SyntheticPlayer.Layout.track.minY) <= 3)
        #expect(progress.frame.width > SyntheticPlayer.Layout.size.width * 0.9)
    }

    @Test(arguments: [0.0, 0.12, 0.5, 0.88]) func everyFractionReadsBack(_ drawn: Double) throws {
        let engine = try VisionEngine()
        let reading = try engine.readMediaControls(in: SyntheticPlayer.image(fraction: drawn))
        let progress = try #require(reading.progress)
        #expect(abs(progress.fraction - drawn) < 0.03)
    }

    /// A PLAYER WITH NO TRANSPORT SHOWING IS NOT A FAILED READ. The controls hide after
    /// a few seconds of no pointer; the answer is `controlsVisible == false`, and the
    /// caller's move is to hover, not to give up.
    @Test func hiddenControlsReadAsHiddenRatherThanAsNothing() throws {
        let engine = try VisionEngine()
        let image = SyntheticPlayer.image(controlsVisible: false)
        let reading = try engine.readMediaControls(in: image)
        #expect(!reading.controlsVisible)
        #expect(reading.progress == nil)
    }

    // MARK: - The controls

    @Test func theControlRowIsFoundLeftToRight() throws {
        let engine = try VisionEngine()
        let image = SyntheticPlayer.image(
            slots: [.play, .next, .volume, .captions, .settings, .fullscreen])
        let reading = try engine.readMediaControls(in: image)

        #expect(reading.controls.count >= 5)
        let xs = reading.controls.map(\.frame.minX)
        #expect(xs == xs.sorted(), "controls must arrive in reading order")
        // A control's frame is its INK, not its hit area — a play triangle is drawn
        // inset inside its slot — so the test that matters is containment.
        let first = try #require(reading.controls.first)
        #expect(SyntheticPlayer.controlRect(at: 0, of: 6).insetBy(dx: -3, dy: -3).contains(first.frame))
    }

    /// THE LEFTMOST CONTROL IS THE TRANSPORT. This is the layout prior the whole lane
    /// rests on, and it is the one that makes an unseen player workable.
    @Test func theTransportIsTheLeftmostControl() throws {
        let engine = try VisionEngine()
        let reading = try engine.readMediaControls(in: SyntheticPlayer.image())
        let transport = try #require(reading.playPause)
        let drawn = SyntheticPlayer.controlRect(at: 0, of: 6)
        #expect(drawn.insetBy(dx: -3, dy: -3).contains(transport.frame))
        // And the point a caller would click lands inside the button it was drawn in.
        #expect(drawn.contains(CGPoint(x: transport.frame.midX, y: transport.frame.midY)))
    }

    @Test func fullscreenIsTheRightmostControl() throws {
        let engine = try VisionEngine()
        let slots: [SyntheticPlayer.Slot] = [.play, .next, .volume, .captions, .settings, .fullscreen]
        let reading = try engine.readMediaControls(in: SyntheticPlayer.image(slots: slots))
        let fullscreen = try #require(reading.fullscreen)
        let drawn = SyntheticPlayer.controlRect(at: slots.count - 1, of: slots.count)
        #expect(abs(fullscreen.frame.midX - drawn.midX) <= 10)
    }

    /// A play triangle and a pause pair must not be read as each other — this is the
    /// one discrimination the base case cannot do without.
    @Test func playAndPauseAreToldApart() throws {
        let engine = try VisionEngine()
        let playing = try engine.readMediaControls(
            in: SyntheticPlayer.image(slots: [.pause, .next, .volume, .captions, .fullscreen]))
        let paused = try engine.readMediaControls(
            in: SyntheticPlayer.image(slots: [.play, .next, .volume, .captions, .fullscreen]))

        let playingGlyph = try #require(playing.playPause).glyph
        let pausedGlyph = try #require(paused.playPause).glyph
        #expect(playingGlyph == .pause)
        #expect(pausedGlyph == .play)
        #expect(playingGlyph.impliesPlaying)
        #expect(pausedGlyph.impliesPaused)
    }

    /// THE PLAYER IS FOUND INSIDE A PAGE, NOT ONLY WHEN IT IS THE WHOLE PICTURE.
    ///
    /// Measured against a real YouTube watch page: the transport sits around 60% of the
    /// way up a 750pt viewport with comments below it, so a scan restricted to the
    /// bottom of the captured region found nothing. This is that regression.
    @Test func theTransportIsFoundPartWayUpAPage() throws {
        let engine = try VisionEngine()
        let page = SyntheticPlayer.watchPage(fraction: 0.4)
        let reading = try engine.readMediaControls(in: page)

        #expect(reading.controlsVisible)
        let progress = try #require(reading.progress)
        #expect(abs(progress.fraction - 0.4) < 0.03)
        // The bar is up where the player is, nowhere near the bottom of the page.
        #expect(progress.frame.midY < 800, "the track is the player's, not the page's")
        #expect(reading.playPause != nil)
    }

    /// A PAGE OF HORIZONTAL RULES IS NOT A PLAYER. Every page has flat two-tone rows;
    /// only a row of buttons underneath makes one a transport.
    @Test func aPageWithNoPlayerReportsNoTransport() throws {
        let engine = try VisionEngine()
        // The same page generator with the player drawn off the top: rules and text only.
        let page = SyntheticPlayer.watchPage(
            fraction: 0.4, playerSize: CGSize(width: 1280, height: 2), pageHeight: 1400)
        let reading = try engine.readMediaControls(in: page)
        #expect(!reading.controlsVisible)
    }

    // MARK: - Motion

    /// TWO FRAMES OF THE SAME PICTURE MEAN NOTHING MOVED. The bar is excluded from the
    /// measurement, so a creeping progress track cannot fake a moving video.
    @Test func aStillPictureReportsNoMotionEvenAsTheTrackMoves() throws {
        let engine = try VisionEngine()
        let before = SyntheticPlayer.image(fraction: 0.30, seed: 7)
        let after = SyntheticPlayer.image(fraction: 0.42, seed: 7)
        let reading = try engine.readMediaControls(in: after, previous: before)

        let motion = try #require(reading.motion)
        #expect(motion <= MediaControlDetection.motionFloor,
                "the picture is identical; only the track moved")
    }

    @Test func aChangedPictureReportsMotion() throws {
        let engine = try VisionEngine()
        let before = SyntheticPlayer.image(fraction: 0.30, seed: 1)
        let after = SyntheticPlayer.image(fraction: 0.32, seed: 2)
        let reading = try engine.readMediaControls(in: after, previous: before)
        let motion = try #require(reading.motion)
        #expect(motion > MediaControlDetection.motionFloor)
    }

    @Test func oneFrameAloneLeavesMotionUnknown() throws {
        let engine = try VisionEngine()
        let reading = try engine.readMediaControls(in: SyntheticPlayer.image())
        #expect(reading.motion == nil)
    }

    // MARK: - The verdict

    /// A moving picture is playing, and the reading SAYS WHICH witness decided it.
    @Test func motionDecidesPlayingAndIsNamed() throws {
        let engine = try VisionEngine()
        let reading = try engine.readMediaControls(
            in: SyntheticPlayer.image(fraction: 0.32, slots: [.pause, .next, .volume, .captions, .fullscreen], seed: 2),
            previous: SyntheticPlayer.image(fraction: 0.30, slots: [.pause, .next, .volume, .captions, .fullscreen], seed: 1))
        #expect(reading.playback == .playing)
        #expect(reading.witnesses.contains { $0.contains("picture moved") })
    }

    /// A still picture with a play glyph is paused, and both witnesses agree.
    @Test func stillnessAndAPlayGlyphAgreeOnPaused() throws {
        let engine = try VisionEngine()
        let frame = SyntheticPlayer.image(
            fraction: 0.30, slots: [.play, .next, .volume, .captions, .fullscreen], seed: 5)
        let reading = try engine.readMediaControls(in: frame, previous: frame)
        #expect(reading.playback == .paused)
        #expect(reading.witnesses.contains { $0.contains("picture still") })
    }

    /// WITH NO WITNESS AT ALL THE ANSWER IS `unknown`, NOT `paused`. One frame of a
    /// player whose controls are hidden proves nothing, and saying "paused" there is
    /// the confident wrong answer this type exists to avoid.
    @Test func noWitnessMeansUnknown() {
        let reading = MediaControlDetection.derive(
            controlsVisible: false, bar: nil, progress: nil, motion: nil,
            controls: [], centerGlyph: nil)
        #expect(reading.playback == .unknown)
        #expect(reading.witnesses.isEmpty)
    }

    /// The progress fraction advancing is a witness in its own right — it is what
    /// proves a video is running when the picture happens to be a static title card.
    @Test func anAdvancingFractionIsEnoughOnItsOwn() {
        let reading = MediaControlDetection.derive(
            controlsVisible: true,
            bar: CGRect(x: 0, y: 640, width: 1280, height: 80),
            progress: .init(frame: CGRect(x: 24, y: 652, width: 1232, height: 5), fraction: 0.42),
            motion: nil, controls: [], centerGlyph: nil,
            previousFraction: 0.30)
        #expect(reading.playback == .playing)
        #expect(reading.witnesses.contains { $0.contains("progress advanced") })
    }

    /// A held fraction is a witness for PAUSED, which is how a seek that did not land
    /// is told from one that did.
    @Test func aHeldFractionWitnessesPaused() {
        let reading = MediaControlDetection.derive(
            controlsVisible: true,
            bar: nil,
            progress: .init(frame: .zero, fraction: 0.30),
            motion: nil, controls: [], centerGlyph: nil,
            previousFraction: 0.30)
        #expect(reading.playback == .paused)
        #expect(reading.witnesses.contains("progress held"))
    }

    /// A HELD SHOT IS NOT A PAUSED VIDEO, AND THE BUTTON KNOWS THE DIFFERENCE.
    ///
    /// Stillness proves nothing on its own — a title card, a held shot and a stopped
    /// video look identical on a coarse grid. Measured on a real page: a running video
    /// during a slow shot read as paused, so a toggle pressed pause and then reported
    /// that nothing had changed. The glyph outranks stillness; it yields to movement.
    @Test func aPauseGlyphOutranksAStillPicture() {
        let bar = CGRect(x: 0, y: 640, width: 1280, height: 80)
        let reading = MediaControlDetection.derive(
            controlsVisible: true, bar: bar, progress: nil, motion: 0.001,
            controls: [.init(frame: CGRect(x: 20, y: 660, width: 24, height: 24),
                             glyph: .pause, confidence: 0.6)],
            centerGlyph: nil)
        #expect(reading.playback == .playing)
        #expect(reading.witnesses.contains { $0.contains("picture still") },
                "the stillness is still reported, it just does not decide")
        #expect(reading.witnesses.contains("pause glyph"))
    }

    /// AND MOVEMENT OUTRANKS THE GLYPH, because a button is one frame behind whatever
    /// was just pressed while a moving picture is happening now.
    @Test func movementOutranksAPlayGlyph() {
        let bar = CGRect(x: 0, y: 640, width: 1280, height: 80)
        let reading = MediaControlDetection.derive(
            controlsVisible: true, bar: bar, progress: nil, motion: 0.4,
            controls: [.init(frame: CGRect(x: 20, y: 660, width: 24, height: 24),
                             glyph: .play, confidence: 0.6)],
            centerGlyph: nil)
        #expect(reading.playback == .playing)
    }

    /// WHEN WITNESSES DISAGREE, MOTION WINS AND THE DISAGREEMENT SURVIVES. A player
    /// showing a static frame of a paused ad while the real video runs behind it is
    /// exactly this case, and a reading that hid it would be unexplainable later.
    @Test func disagreementKeepsBothWitnesses() {
        let reading = MediaControlDetection.derive(
            controlsVisible: true, bar: nil,
            progress: .init(frame: .zero, fraction: 0.30),
            motion: 0.4, controls: [], centerGlyph: nil,
            previousFraction: 0.30)
        #expect(reading.playback == .playing)
        #expect(reading.witnesses.contains { $0.contains("picture moved") })
        #expect(reading.witnesses.contains("progress held"))
    }

    // MARK: - Mute

    @Test func aSpeakerWithACrossReadsAsMuted() throws {
        let engine = try VisionEngine()
        let muted = try engine.readMediaControls(
            in: SyntheticPlayer.image(slots: [.play, .muted, .captions, .settings, .fullscreen]))
        let loud = try engine.readMediaControls(
            in: SyntheticPlayer.image(slots: [.play, .volume, .captions, .settings, .fullscreen]))
        #expect(muted.isMuted == true)
        #expect(loud.isMuted == false)
    }

    /// NO VOLUME CONTROL MEANS UNKNOWN, NOT UNMUTED. A player that hides its speaker
    /// behind an overflow menu has not told us the sound is on.
    @Test func noVolumeControlLeavesMuteUnknown() throws {
        let engine = try VisionEngine()
        let reading = try engine.readMediaControls(
            in: SyntheticPlayer.image(slots: [.play, .next, .captions, .settings, .fullscreen]))
        #expect(reading.isMuted == nil)
    }
}
