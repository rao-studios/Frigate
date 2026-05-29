// Linux compatibility shim — provides types that are Darwin-only on Apple platforms.

#if !canImport(CoreGraphics)
public struct CGSize: Sendable, Hashable {
    public var width: Double
    public var height: Double
    public init(width: Double, height: Double) {
        self.width = width
        self.height = height
    }
}
#endif
