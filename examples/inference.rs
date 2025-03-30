use argh::FromArgs;
use std::path::PathBuf;

use kornia_io::functional as F;
use kornia_yolo::{YoloV8, YoloV8Config, YoloV8Size};

#[derive(FromArgs)]
/// YOLOv8 inference application arguments
struct Args {
    /// path to an input image
    #[argh(option)]
    image_path: PathBuf,

    /// the size of the model to use: n, s, m, l, x
    #[argh(option, default = "\"n\".to_string()")]
    size: String,

    /// the confidence threshold for the model
    #[argh(option, default = "0.25")]
    confidence_threshold: f32,

    /// the nms threshold for the model
    #[argh(option, default = "0.5")]
    nms_threshold: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Args = argh::from_env();

    let rec = rerun::RecordingStreamBuilder::new("YOLOv8 Inference App").spawn()?;

    let config = YoloV8Config {
        size: YoloV8Size::try_from(args.size)?,
        confidence_threshold: args.confidence_threshold,
        nms_threshold: args.nms_threshold,
        use_cpu: true,
    };

    let model = YoloV8::new(config)?;

    // read the image as RGB8
    let image = F::read_image_any(args.image_path)?;

    // perform inference and get the detections
    let detections = model.inference(&image)?;

    rec.log(
        "image",
        &rerun::Image::from_elements(
            image.as_slice(),
            image.size().into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    let mut boxes_mins = Vec::new();
    let mut boxes_sizes = Vec::new();
    let mut labels = Vec::new();
    for detection in detections {
        boxes_mins.push((detection.xmin, detection.ymin));
        boxes_sizes.push((
            detection.xmax - detection.xmin,
            detection.ymax - detection.ymin,
        ));
        labels.push(detection.class as u16);
    }

    rec.log(
        "boxes",
        &rerun::Boxes2D::from_mins_and_sizes(boxes_mins, boxes_sizes).with_class_ids(labels),
    )?;

    Ok(())
}
