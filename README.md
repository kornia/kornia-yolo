# Kornia YOLO in Rust

A fast, reliable YOLO object detection implementation for Rust powered by Kornia and Candle.

## Features

- 🚀 High-performance object detection with YOLOv8 using Kornia and Candle
- 🔧 Multiple model sizes (n, s, m, l, x) for different performance/accuracy tradeoffs
- 📦 Pre-trained models automatically downloaded from Hugging Face
- 🖼️ Simple API for inference on images
- 🧮 Non-maximum suppression for refined detection results
- 🔄 Built on the Kornia Rust computer vision ecosystem

## Installation

```bash
[dependencies]
kornia-yolo = { git = "https://github.com/kornia/kornia-yolo", tag = "0.1.0" }
```

## Quickstart

```rust
use kornia_io::functional as F;
use kornia_yolo::{YoloV8, YoloV8Config, YoloV8Size};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a YOLOv8 model with default configuration (nano size)
    let model = YoloV8::new(YoloV8Config::default())?;
    
    // Load an image
    let image = F::read_image_any("path/to/your/image.jpg")?;
    
    // Run inference
    let detections = model.inference(&image)?;
    
    // Process and display results
    for detection in detections {
        println!(
            "Detected: class={}, confidence={:.2}, bbox=({:.1}, {:.1}, {:.1}, {:.1})",
            detection.class,
            detection.confidence,
            detection.xmin, detection.ymin, detection.xmax, detection.ymax
        );
    }
    
    Ok(())
}
```

## Example

```bash
cargo run --example inference --release -- --image_path ./tests/data/bike.jpg
```

## Acknowledgements

This project is built on top of the following amazing projects:

- [Kornia](https://github.com/kornia/kornia-rs)
- [Candle](https://github.com/huggingface/candle)
- [Candle-Yolov8](https://github.com/huggingface/candle/tree/main/candle-examples/examples/yolo-v8)
- [Tinygrad-Yolov8](https://github.com/tinygrad/tinygrad/blob/master/examples/yolov8.py)

## License

This project has an uncleared license as it is a derivative work of the above projects including a modified version of the YOLOv8 model from [Ultralytics](https://github.com/ultralytics/ultralytics).

## Contributing

This is a child project of Kornia. Join the community to get in touch with us, or just sponsor the project: https://opencollective.com/kornia