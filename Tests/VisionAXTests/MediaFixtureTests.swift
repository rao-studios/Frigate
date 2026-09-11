//
//  MediaFixtureTests.swift
//  VisionAXTests
//
//  WHAT: The media lane against REAL captures of real players.
//  OUT:  Fixtures/media/*.png
//  PIN:  A DRAWN PLAYER PROVES THE PIPELINE; ONLY A REAL ONE PROVES THE PIPELINE WORKS.
//        The synthetic fixtures next door exercise the band scan and the witnesses
//        against known answers, and every one of them passed while the detector could
//        not find YouTube's transport at all. These captures are the other half.
//        Set VISIONAX_MEDIA_DUMP=1 to print what each candidate scored — the loop this
//        was tuned in, kept because the next player will need it too.
//

import CoreGraphics
import Foundation
import ImageIO
import Testing
@testable import VisionAX

@Suite struct MediaFixtureTests {

    /// A capture off disk, or nil when the working tree has only its git-lfs pointer.
    ///
    /// PIN: AN UN-PULLED FIXTURE IS A CONFIGURATION, NOT A FAILURE. These are LFS
    /// objects; a clone without `git lfs pull` leaves 130 bytes of text where a
    /// screenshot should be, and letting the image decoder fail on that reports a broken
    /// detector to whoever is reading the suite. Detected by name here, and said out
    /// loud — the same courtesy `RegionClassifier` pays a missing model.
    static func image(named name: String) throws -> CGImage? {
        let url = try #require(
            Bundle.module.url(forResource: name, withExtension: "png", subdirectory: "Fixtures/media"),
            "fixture \(name).png is missing")
        let head = try Data(contentsOf: url).prefix(48)
        if head.starts(with: Data("version https://git-lfs.github.com/spec".utf8)) {
            return nil
        }
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        return try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
    }

    /// A YouTube watch page in Safari, controls revealed, video paused at 0:37 of 10:34.
    /// Captured live at 1x through the browser lane's own capture path.
    @Test func youTubeInSafariIsRead() throws {
        let engine = try VisionEngine()
        guard let page = try Self.image(named: "youtube-safari-paused") else { return }
        let reading = try engine.readMediaControls(in: page)

        if ProcessInfo.processInfo.environment["VISIONAX_MEDIA_DUMP"] != nil {
            print("bar \(String(describing: reading.bar))")
            print("progress \(String(describing: reading.progress))")
            for control in reading.controls {
                print("  control \(control.frame) \(control.glyph) \(control.confidence)")
            }
            // The player alone, to separate "the row scan cannot see it" from
            // "something else outscored it".
            let scene = try engine.perceive(
                image: page, projection: .imageSpace,
                region: CGRect(x: 0, y: 0, width: 900, height: 580), lanes: [.media])
            print("PLAYER-ONLY bar \(String(describing: scene.media?.bar))")
            print("PLAYER-ONLY progress \(String(describing: scene.media?.progress))")
            for control in scene.media?.controls ?? [] {
                print("  player control \(control.frame) \(control.glyph)")
            }
        }

        #expect(reading.controlsVisible, "the transport is plainly in the picture")
        let progress = try #require(reading.progress)
        // 0:37 of 10:34 is 5.8% — the red run ends about a sixteenth of the way across.
        #expect(abs(progress.fraction - 0.058) < 0.03)
        // The bar sits around two thirds down the 752pt page, NOT at its bottom.
        #expect(progress.frame.midY > 480 && progress.frame.midY < 540)

        let transport = try #require(reading.playPause)
        // Play is the leftmost control, near the left edge of the player.
        #expect(transport.frame.midX < 80)
        #expect(transport.frame.midY > 520 && transport.frame.midY < 570)

        let fullscreen = try #require(reading.fullscreen)
        #expect(fullscreen.frame.midX > 800, "full screen is the rightmost control")
    }

    /// The same page while PLAYING, with the transport revealed by a hover.
    ///
    /// The paused capture is the easy one — nothing moves and the big centred glyph
    /// corroborates. This is the case that matters: a playing video hides its controls
    /// until the pointer moves over it, and the frame that comes back has the transport
    /// drawn over MOVING pixels rather than a still.
    @Test func youTubeInSafariPlayingIsRead() throws {
        let engine = try VisionEngine()
        guard let page = try Self.image(named: "youtube-safari-playing") else { return }
        let reading = try engine.readMediaControls(in: page)

        if ProcessInfo.processInfo.environment["VISIONAX_MEDIA_DUMP"] != nil {
            print("PLAYING bar \(String(describing: reading.bar))")
            print("PLAYING progress \(String(describing: reading.progress))")
            for control in reading.controls {
                print("  control \(control.frame) \(control.glyph) \(control.confidence)")
            }
        }

        #expect(reading.controlsVisible)
        let progress = try #require(reading.progress)
        // The transport is two thirds down the page, where the player's bottom edge is —
        // NOT at the masthead, and not at the metadata row below the video.
        #expect(progress.frame.midY > 480 && progress.frame.midY < 545,
                "the track is the player's, not the page's")
        // 0:39 of 10:34 is 6.2%.
        #expect(abs(progress.fraction - 0.062) < 0.035)

        let transport = try #require(reading.playPause)
        #expect(transport.frame.midX < 80, "play/pause is at the left end of the bar")
        // THE BUTTON SAYS PAUSE, because pressing it would pause: the video is running.
        #expect(transport.glyph == .pause)
    }

    /// The same player nine seconds into a ten-minute video.
    ///
    /// The played portion is about twelve pixels of a nine-hundred pixel bar, which is
    /// SHORTER THAN THE DETECTOR'S OWN IDEA OF A RUN. Without reaching back over it the
    /// bar reads as one flat unplayed line — indistinguishable from a page divider — and
    /// a plainly visible transport was reported as no transport at all.
    @Test func aBarelyStartedVideoStillHasATransport() throws {
        let engine = try VisionEngine()
        guard let page = try Self.image(named: "youtube-safari-just-started") else { return }
        let reading = try engine.readMediaControls(in: page)

        #expect(reading.controlsVisible)
        let progress = try #require(reading.progress)
        // 0:09 of 10:34 is 1.4%; the nub is wider than the position it represents.
        #expect(progress.fraction < 0.06)
        #expect(progress.frame.midY > 490 && progress.frame.midY < 545)

        let transport = try #require(reading.playPause)
        #expect(transport.glyph == .play)
        #expect(transport.frame.midX < 60)
        #expect(try #require(reading.fullscreen).frame.midX > 800)
    }

    /// THE OTHER BROWSER, AND A DIFFERENT PLAYER LAYOUT.
    ///
    /// Chrome renders YouTube's newer control bar: the right-hand buttons sit inside a
    /// rounded pill, the picture is letterboxed inside a wider player, and the unplayed
    /// track carries more of the video's variation than Safari's does. Nothing about the
    /// lane is per-browser, and this fixture is what keeps that true.
    @Test func youTubeInChromeIsRead() throws {
        let engine = try VisionEngine()
        guard let page = try Self.image(named: "youtube-chrome-playing") else { return }
        let reading = try engine.readMediaControls(in: page)

        if ProcessInfo.processInfo.environment["VISIONAX_MEDIA_DUMP"] != nil {
            print("CHROME bar \(String(describing: reading.bar))")
            print("CHROME progress \(String(describing: reading.progress))")
            for control in reading.controls {
                print("  control \(control.frame) \(control.glyph) \(control.confidence)")
            }
        }

        #expect(reading.controlsVisible)
        let progress = try #require(reading.progress)
        #expect(progress.frame.midY > 490 && progress.frame.midY < 530)
        // 0:31 of 10:34 is 4.9%.
        #expect(abs(progress.fraction - 0.049) < 0.035)

        let transport = try #require(reading.playPause)
        #expect(transport.glyph == .pause, "the video is running, so the button offers pause")
        #expect(transport.frame.midX < 70)
        #expect(try #require(reading.fullscreen).frame.midX > 1000)
    }

    /// THE HARDEST FRAME SO FAR: Chrome, paused over a sunlit meadow.
    ///
    /// The scrim is a gradient over a BRIGHT picture, which defeated a single brightness
    /// threshold — the grass came through as blobs that swallowed the play button, and a
    /// plainly visible transport was refused. It is also the case the flat-run scan
    /// cannot span, because a translucent unplayed track over moving grass is not flat
    /// anywhere; the accent run carries it instead.
    @Test func youTubeInChromeOverABrightPictureIsRead() throws {
        let engine = try VisionEngine()
        guard let page = try Self.image(named: "youtube-chrome-paused-bright") else { return }
        let reading = try engine.readMediaControls(in: page)

        #expect(reading.controlsVisible)
        let progress = try #require(reading.progress)
        // 2:50 of 10:34 is 26.8%.
        #expect(abs(progress.fraction - 0.268) < 0.035)
        #expect(progress.frame.midY > 490 && progress.frame.midY < 525)

        let transport = try #require(reading.playPause)
        #expect(transport.glyph == .play)
        #expect(transport.frame.midX < 60)
    }

    /// THE HARD FRAME, AND THE INVARIANT THAT MATTERS ON IT.
    ///
    /// White glyphs over the brightest frame of a video is the case this lane is worst
    /// at: the play triangle sits on a sunlit picture, so it is neither brighter than its
    /// neighbourhood nor distinguishable by an absolute level, and the transport is
    /// sometimes not found at all. That is a real limit and it is written down in
    /// docs/browser-engine.md.
    ///
    /// What must NEVER happen is the other failure: reporting a transport somewhere else
    /// on the page. A refusal costs a retry; a wrong transport costs a click on
    /// somebody's Subscribe button. This pins the difference.
    @Test func aBrightFrameRefusesRatherThanInventingATransport() throws {
        let engine = try VisionEngine()
        guard let page = try Self.image(named: "youtube-safari-bright-frame") else { return }
        let reading = try engine.readMediaControls(in: page)

        guard let progress = reading.progress else {
            #expect(!reading.controlsVisible, "no track means no transport was claimed")
            return
        }
        // If it DID find one, it is the player's own bar and nothing else.
        #expect(progress.frame.midY > 490 && progress.frame.midY < 545)
        let transport = try #require(reading.playPause)
        #expect(transport.frame.midX < 80)
    }

    /// The clock in the bar reads back, and nothing outside it does.
    @Test(arguments: [
        ("youtube-safari-paused", 37.0, 634.0),
        ("youtube-safari-just-started", 9.0, 634.0),
        ("youtube-chrome-playing", 31.0, 634.0),
        ("youtube-chrome-paused-bright", 170.0, 634.0),
    ]) func theClockIsReadFromTheBar(_ fixture: String, _ elapsed: Double, _ duration: Double) throws {
        let engine = try VisionEngine()
        guard let page = try Self.image(named: fixture) else { return }
        let scene = try engine.perceive(
            image: page, projection: .imageSpace, lanes: [.media, .text])
        let media = try #require(scene.media)
        #expect(media.elapsed == elapsed)
        #expect(media.duration == duration)
    }
}
