// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "JustSayAudioTap",
    platforms: [.macOS("14.4")],
    products: [
        .executable(name: "justsay-audiotap", targets: ["justsay-audiotap"])
    ],
    targets: [
        .executableTarget(
            name: "justsay-audiotap",
            path: "Sources/JustSayAudioTap"
        )
    ]
)
