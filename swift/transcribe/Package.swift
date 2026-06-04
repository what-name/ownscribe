// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ownscribe-transcribe",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.14.0"),
    ],
    targets: [
        .executableTarget(
            name: "ownscribe-transcribe",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            path: "Sources"
        )
    ]
)
