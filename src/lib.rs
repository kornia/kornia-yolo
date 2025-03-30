#![deny(missing_docs)]

//! YOLOv8 inference in Rust
//!
//! This crate provides a simple interface for running YOLOv8 inference on images.
//!
//! # Examples
//!
//! ```no_run
//! use kornia_yolo::{YoloV8, YoloV8Config, YoloV8Size};
//!
//! let config = YoloV8Config {
//!     size: YoloV8Size::N,
//!     confidence_threshold: 0.25,
//!     nms_threshold: 0.45,
//!     use_cpu: true,
//! };
//!
//! let model = YoloV8::new(config).expect("Failed to create YOLOv8 model");
//!
//! let image = kornia_io::functional::read_image_any("path/to/image.jpg")
//!     .expect("Failed to read image");
//!
//! let detections = model.inference(&image).expect("Failed to run inference");
//! for detection in detections {
//!     println!("Detection: {:?}", detection);
//! }
//! ```

/// Bounding box module with non-maximum suppression
mod bounding_box;

/// YOLOv8 model definition in candle
mod model;

/// YOLOv8 high level interface
mod yolov8;

pub use bounding_box::{BoundingBox, non_maximum_suppression};
pub use yolov8::{YoloV8, YoloV8Config, YoloV8Error, YoloV8Size};
