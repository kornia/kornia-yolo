use super::{
    bounding_box::{BoundingBox, non_maximum_suppression},
    model::{Multiples, YoloV8 as YoloV8Model},
};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Module, VarBuilder};
use kornia_image::{Image, ImageSize};

/// YOLOv8 error enum.
#[derive(thiserror::Error, Debug)]
pub enum YoloV8Error {
    /// Failed to load YOLOv8 model.
    #[error("Failed to load YOLOv8 model: {0}")]
    LoadModelError(String),

    /// Internal candle error.
    #[error(transparent)]
    CandleError(#[from] candle_core::Error),

    /// Kornia image error.
    #[error(transparent)]
    KorniaImageError(#[from] kornia_image::ImageError),

    /// Hugging Face API error.
    #[error(transparent)]
    ApiError(#[from] hf_hub::api::sync::ApiError),
}

/// YOLOv8 model size enum.
pub enum YoloV8Size {
    /// N model size.
    N,
    /// S model size.
    S,
    /// M model size.
    M,
    /// L model size.
    L,
    /// X model size.
    X,
}

impl TryFrom<String> for YoloV8Size {
    type Error = YoloV8Error;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        Ok(match s.as_str() {
            "n" => YoloV8Size::N,
            "s" => YoloV8Size::S,
            "m" => YoloV8Size::M,
            "l" => YoloV8Size::L,
            "x" => YoloV8Size::X,
            _ => return Err(YoloV8Error::LoadModelError(format!("Invalid size: {s}"))),
        })
    }
}

impl std::fmt::Display for YoloV8Size {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                YoloV8Size::N => "n",
                YoloV8Size::S => "s",
                YoloV8Size::M => "m",
                YoloV8Size::L => "l",
                YoloV8Size::X => "x",
            }
        )
    }
}

/// YOLOv8 configuration struct.
pub struct YoloV8Config {
    /// The size of the model to use: n, s, m, l, x
    pub size: YoloV8Size,
    /// The confidence threshold for the model
    pub confidence_threshold: f32,
    /// The non-maximum suppression threshold for the model
    pub nms_threshold: f32,
    /// Whether to use the CPU or GPU
    pub use_cpu: bool,
}

/// Default configuration for YOLOv8.
impl Default for YoloV8Config {
    fn default() -> Self {
        Self {
            size: YoloV8Size::N,
            confidence_threshold: 0.25,
            nms_threshold: 0.45,
            use_cpu: true,
        }
    }
}

/// YOLOv8 high level interface.
pub struct YoloV8 {
    config: YoloV8Config,
    device: Device,
    model: YoloV8Model,
}

impl YoloV8 {
    /// Create a new YOLOv8 instance
    pub fn new(config: YoloV8Config) -> Result<Self, YoloV8Error> {
        let device = if config.use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)
                .map_err(|_| YoloV8Error::LoadModelError("cuda:0 error".to_string()))?
        };
        let model = Self::load_model(&config.size, &device)?;
        Ok(Self {
            config,
            device,
            model,
        })
    }

    /// Perform inference on an rgb8 image
    pub fn inference(&self, image: &Image<u8, 3>) -> Result<Vec<BoundingBox>, YoloV8Error> {
        // preprocess the image
        let (image_t, w_ratio, h_ratio) = self.preprocess_image(image)?;

        // forward the image
        let pred = self.model.forward(&image_t)?.squeeze(0)?;

        // postprocess the predictions
        self.postprocess_predictions(&pred, w_ratio, h_ratio)
    }

    fn preprocess_image(&self, image: &Image<u8, 3>) -> Result<(Tensor, f32, f32), YoloV8Error> {
        let (width, height) = {
            let w = image.width();
            let h = image.height();
            if w < h {
                let w = w * 640 / h;
                // Sizes have to be divisible by 32.
                (w / 32 * 32, 640)
            } else {
                let h = h * 640 / w;
                (640, h / 32 * 32)
            }
        };

        let w_ratio = image.width() as f32 / width as f32;
        let h_ratio = image.height() as f32 / height as f32;

        let mut image_resized = Image::from_size_val(ImageSize { width, height }, 0)?;
        kornia_imgproc::resize::resize_fast(
            image,
            &mut image_resized,
            kornia_imgproc::interpolation::InterpolationMode::Nearest,
        )?;

        let image_resized = image_resized.map(|&x| x as f32 / 255.0);

        let image_t =
            Tensor::from_vec::<_, f32>(image_resized.into_vec(), (height, width, 3), &self.device)?
                .permute((2, 0, 1))?
                .unsqueeze(0)?;

        Ok((image_t, w_ratio, h_ratio))
    }

    fn postprocess_predictions(
        &self,
        pred: &Tensor,
        w_ratio: f32,
        h_ratio: f32,
    ) -> Result<Vec<BoundingBox>, YoloV8Error> {
        let (pred_size, npreds) = pred.dims2()?;
        let nclasses = pred_size - 4;
        let mut bboxes: Vec<Vec<BoundingBox>> = (0..nclasses).map(|_| Vec::new()).collect();
        for index in 0..npreds {
            let pred = Vec::<f32>::try_from(pred.i((.., index))?)?;
            let confidence = *pred[4..].iter().max_by(|x, y| x.total_cmp(y)).unwrap();
            if confidence > self.config.confidence_threshold {
                let mut class_index = 0;
                for i in 0..nclasses {
                    if pred[4 + i] > pred[4 + class_index] {
                        class_index = i;
                    }
                }
                if pred[4 + class_index] > 0. {
                    let bbox = BoundingBox {
                        xmin: (pred[0] - pred[2] / 2.0) * w_ratio,
                        ymin: (pred[1] - pred[3] / 2.0) * h_ratio,
                        xmax: (pred[0] + pred[2] / 2.0) * w_ratio,
                        ymax: (pred[1] + pred[3] / 2.0) * h_ratio,
                        confidence,
                        class: class_index as u32,
                    };
                    bboxes[class_index].push(bbox);
                }
            }
        }

        // non-maximum suppression
        // TODO: implement faster version from Thomas M.
        non_maximum_suppression(&mut bboxes, self.config.nms_threshold);

        Ok(bboxes.into_iter().flatten().collect())
    }

    fn load_model(size: &YoloV8Size, device: &Device) -> Result<YoloV8Model, YoloV8Error> {
        // TODO: add support for other custom models
        // check if the model is already downloaded or download it
        let model_path = hf_hub::api::sync::Api::new()?
            .model("lmz/candle-yolo-v8".to_string())
            .get(&format!("yolov8{}.safetensors", size))?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, device)? };
        let m = match size {
            YoloV8Size::N => Multiples::n(),
            YoloV8Size::S => Multiples::s(),
            YoloV8Size::M => Multiples::m(),
            YoloV8Size::L => Multiples::l(),
            YoloV8Size::X => Multiples::x(),
        };

        Ok(YoloV8Model::load(vb, m, /* num_classes */ 80)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_io::functional as F;

    const TEST_IMAGE: &str = "tests/data/bike.jpg";

    #[test]
    fn test_yolov8_inference() -> Result<(), Box<dyn std::error::Error>> {
        let yolov8 = YoloV8::new(YoloV8Config::default())?;
        let image = F::read_image_any(TEST_IMAGE)?;
        let detections = yolov8.inference(&image)?;
        assert!(!detections.is_empty());
        Ok(())
    }
}
