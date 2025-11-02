# visagist

100% local AI meme detection. No uploads, no tracking, no servers.

Drop images, get instant meme analysis powered by a neural network running entirely in your browser.

## Features

- **Instant detection** - Sub-second classification using WebGPU acceleration
- **Complete privacy** - All processing happens locally, images never leave your device
- **Smart algorithm** - Handles uncertain cases by defaulting to meme classification
- **Batch processing** - Analyze multiple images simultaneously
- **Export results** - Download analysis data as JSON

## Quick Start

```bash
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) and start dropping images.

## Build

```bash
pnpm build
```

Deploy the `dist` folder to any static host.

## Tech Stack

- **SolidJS** - Reactive UI framework
- **Transformers.js** - In-browser ML inference
- **WebGPU/WASM** - Hardware acceleration fallback
- **Custom ONNX model** - Fine-tuned meme classifier
