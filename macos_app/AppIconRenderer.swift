import AppKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

private struct IconEntry {
    let filename: String
    let pixels: Int
}

private struct ICNSChunk {
    let type: Data
    let payload: Data
}

private enum RendererError: LocalizedError {
    case invalidArguments
    case unreadableSVG(String)
    case bitmapCreationFailed(Int)
    case pngEncodingFailed(Int)
    case cgImageCreationFailed(Int)
    case tiffDestinationCreationFailed
    case tiffEncodingFailed
    case invalidICNS(String)

    var errorDescription: String? {
        switch self {
        case .invalidArguments:
            return "用法：AppIconRenderer <AppIcon.svg> <AppIcon.iconset> <AppIcon.tiff>，" +
                "或 AppIconRenderer --add-1024 <1024.png> <AppIcon.icns>"
        case let .unreadableSVG(path):
            return "无法读取 SVG：\(path)"
        case let .bitmapCreationFailed(pixels):
            return "无法创建 \(pixels)x\(pixels) 位图"
        case let .pngEncodingFailed(pixels):
            return "无法编码 \(pixels)x\(pixels) PNG"
        case let .cgImageCreationFailed(pixels):
            return "无法创建 \(pixels)x\(pixels) CGImage"
        case .tiffDestinationCreationFailed:
            return "无法创建多分辨率 TIFF"
        case .tiffEncodingFailed:
            return "无法写入多分辨率 TIFF"
        case let .invalidICNS(path):
            return "无效的 ICNS 文件：\(path)"
        }
    }
}

private let iconEntries = [
    IconEntry(filename: "icon_16x16.png", pixels: 16),
    IconEntry(filename: "icon_16x16@2x.png", pixels: 32),
    IconEntry(filename: "icon_32x32.png", pixels: 32),
    IconEntry(filename: "icon_32x32@2x.png", pixels: 64),
    IconEntry(filename: "icon_128x128.png", pixels: 128),
    IconEntry(filename: "icon_128x128@2x.png", pixels: 256),
    IconEntry(filename: "icon_256x256.png", pixels: 256),
    IconEntry(filename: "icon_256x256@2x.png", pixels: 512),
    IconEntry(filename: "icon_512x512.png", pixels: 512),
    IconEntry(filename: "icon_512x512@2x.png", pixels: 1024),
]

private func appendFourCC(_ value: String, to data: inout Data) {
    data.append(contentsOf: value.utf8)
}

private func appendUInt32(_ value: Int, to data: inout Data) {
    var bigEndianValue = UInt32(value).bigEndian
    withUnsafeBytes(of: &bigEndianValue) { bytes in
        data.append(contentsOf: bytes)
    }
}

private func readUInt32(from data: Data, at offset: Int) -> Int {
    data[offset..<(offset + 4)].reduce(0) { value, byte in
        (value << 8) | Int(byte)
    }
}

private func add1024PNG(_ pngURL: URL, to icnsURL: URL) throws {
    let icnsData = try Data(contentsOf: icnsURL)
    let pngData = try Data(contentsOf: pngURL)
    let icnsMagic = Data("icns".utf8)
    let tocType = Data("TOC ".utf8)
    let icon1024Type = Data("ic10".utf8)

    guard icnsData.count >= 8, icnsData.prefix(4) == icnsMagic else {
        throw RendererError.invalidICNS(icnsURL.path)
    }

    var chunks: [ICNSChunk] = []
    var offset = 8
    while offset + 8 <= icnsData.count {
        let type = icnsData.subdata(in: offset..<(offset + 4))
        let chunkSize = readUInt32(from: icnsData, at: offset + 4)
        guard chunkSize >= 8, offset + chunkSize <= icnsData.count else {
            throw RendererError.invalidICNS(icnsURL.path)
        }

        if type != tocType, type != icon1024Type {
            chunks.append(
                ICNSChunk(
                    type: type,
                    payload: icnsData.subdata(in: (offset + 8)..<(offset + chunkSize))
                )
            )
        }
        offset += chunkSize
    }

    guard offset == icnsData.count else {
        throw RendererError.invalidICNS(icnsURL.path)
    }

    chunks.append(ICNSChunk(type: icon1024Type, payload: pngData))
    let tocSize = 8 + chunks.count * 8
    let totalSize = 8 + tocSize + chunks.reduce(0) { size, chunk in
        size + 8 + chunk.payload.count
    }

    var output = Data()
    appendFourCC("icns", to: &output)
    appendUInt32(totalSize, to: &output)
    appendFourCC("TOC ", to: &output)
    appendUInt32(tocSize, to: &output)
    for chunk in chunks {
        output.append(chunk.type)
        appendUInt32(8 + chunk.payload.count, to: &output)
    }
    for chunk in chunks {
        output.append(chunk.type)
        appendUInt32(8 + chunk.payload.count, to: &output)
        output.append(chunk.payload)
    }

    try output.write(to: icnsURL, options: .atomic)
}

private func renderBitmap(from sourceImage: NSImage, pixels: Int) throws -> NSBitmapImageRep {
    guard let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: pixels,
        pixelsHigh: pixels,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    ) else {
        throw RendererError.bitmapCreationFailed(pixels)
    }

    bitmap.size = NSSize(width: pixels, height: pixels)
    guard let context = NSGraphicsContext(bitmapImageRep: bitmap) else {
        throw RendererError.bitmapCreationFailed(pixels)
    }

    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = context
    context.imageInterpolation = .high
    NSColor.clear.setFill()
    NSRect(x: 0, y: 0, width: pixels, height: pixels).fill()
    sourceImage.draw(
        in: NSRect(x: 0, y: 0, width: pixels, height: pixels),
        from: NSRect(origin: .zero, size: sourceImage.size),
        operation: .sourceOver,
        fraction: 1,
        respectFlipped: true,
        hints: [.interpolation: NSImageInterpolation.high]
    )
    context.flushGraphics()
    NSGraphicsContext.restoreGraphicsState()

    return bitmap
}

private func renderIcon(from svgURL: URL, to outputDirectory: URL, tiffURL: URL) throws {
    guard let sourceImage = NSImage(contentsOf: svgURL) else {
        throw RendererError.unreadableSVG(svgURL.path)
    }

    let fileManager = FileManager.default
    try fileManager.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
    try fileManager.createDirectory(at: tiffURL.deletingLastPathComponent(), withIntermediateDirectories: true)

    for entry in iconEntries {
        let bitmap = try renderBitmap(from: sourceImage, pixels: entry.pixels)
        guard let pngData = bitmap.representation(using: .png, properties: [:]) else {
            throw RendererError.pngEncodingFailed(entry.pixels)
        }
        try pngData.write(to: outputDirectory.appendingPathComponent(entry.filename), options: .atomic)
    }

    let tiffSizes = [48, 32, 16, 128, 256, 512, 1024]
    guard let destination = CGImageDestinationCreateWithURL(
        tiffURL as CFURL,
        UTType.tiff.identifier as CFString,
        tiffSizes.count,
        nil
    ) else {
        throw RendererError.tiffDestinationCreationFailed
    }

    for pixels in tiffSizes {
        let bitmap = try renderBitmap(from: sourceImage, pixels: pixels)
        guard let cgImage = bitmap.cgImage else {
            throw RendererError.cgImageCreationFailed(pixels)
        }
        CGImageDestinationAddImage(destination, cgImage, nil)
    }

    guard CGImageDestinationFinalize(destination) else {
        throw RendererError.tiffEncodingFailed
    }
}

do {
    guard CommandLine.arguments.count == 4 else {
        throw RendererError.invalidArguments
    }

    if CommandLine.arguments[1] == "--add-1024" {
        let pngURL = URL(fileURLWithPath: CommandLine.arguments[2])
        let icnsURL = URL(fileURLWithPath: CommandLine.arguments[3])
        try add1024PNG(pngURL, to: icnsURL)
    } else {
        let svgURL = URL(fileURLWithPath: CommandLine.arguments[1])
        let outputDirectory = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
        let tiffURL = URL(fileURLWithPath: CommandLine.arguments[3])
        try renderIcon(from: svgURL, to: outputDirectory, tiffURL: tiffURL)
    }
} catch {
    fputs("错误：\(error.localizedDescription)\n", stderr)
    exit(1)
}
