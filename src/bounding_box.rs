use serde::Serialize;

/// Bounding box struct for detection results.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct BoundingBox {
    /// Minimum x coordinate of the bounding box.
    pub xmin: f32,
    /// Minimum y coordinate of the bounding box.
    pub ymin: f32,
    /// Maximum x coordinate of the bounding box.
    pub xmax: f32,
    /// Maximum y coordinate of the bounding box.
    pub ymax: f32,
    /// Confidence score of the bounding box.
    pub confidence: f32,
    /// Class index of the bounding box.
    pub class: u32,
}

/// Intersection over union of two bounding boxes.
fn iou(b1: &BoundingBox, b2: &BoundingBox) -> f32 {
    let b1_area = (b1.xmax - b1.xmin + 1.) * (b1.ymax - b1.ymin + 1.);
    let b2_area = (b2.xmax - b2.xmin + 1.) * (b2.ymax - b2.ymin + 1.);
    let i_xmin = b1.xmin.max(b2.xmin);
    let i_xmax = b1.xmax.min(b2.xmax);
    let i_ymin = b1.ymin.max(b2.ymin);
    let i_ymax = b1.ymax.min(b2.ymax);
    let i_area = (i_xmax - i_xmin + 1.).max(0.) * (i_ymax - i_ymin + 1.).max(0.);
    i_area / (b1_area + b2_area - i_area)
}

/// Non-maximum suppression for bounding boxes.
///
/// This function performs non-maximum suppression on a list of bounding boxes.
/// It removes overlapping boxes with a IoU greater than the threshold.
///
/// # Arguments
///
/// * `bboxes` - A mutable reference to a vector of bounding boxes.
/// * `threshold` - The IoU threshold for suppression.
pub fn non_maximum_suppression(bboxes: &mut [Vec<BoundingBox>], threshold: f32) {
    // Perform non-maximum suppression.
    for bboxes_for_class in bboxes.iter_mut() {
        bboxes_for_class.sort_by(|b1, b2| b2.confidence.partial_cmp(&b1.confidence).unwrap());
        let mut current_index = 0;
        for index in 0..bboxes_for_class.len() {
            let mut drop = false;
            for prev_index in 0..current_index {
                let iou = iou(&bboxes_for_class[prev_index], &bboxes_for_class[index]);
                if iou > threshold {
                    drop = true;
                    break;
                }
            }
            if !drop {
                bboxes_for_class.swap(current_index, index);
                current_index += 1;
            }
        }
        bboxes_for_class.truncate(current_index);
    }
}

/// Non-maximum suppression for bounding boxes.
///
/// Fast implementation from:
/// https://www.computervisionblog.com/2011/08/blazing-fast-nmsm-from-exemplar-svm.html
///
/// Args:
///     boxes: A mutable reference to a vector of bounding boxes.
///     threshold: The IoU threshold for suppression.
///
/// Returns:
///     A vector of bounding boxes that were picked.
pub fn non_maximum_suppression_fast(boxes: &[BoundingBox], threshold: f32) -> Vec<BoundingBox> {
    if boxes.is_empty() {
        return vec![];
    }

    // Sort indices by confidence (descending)
    let mut idxs: Vec<usize> = (0..boxes.len()).collect();
    idxs.sort_by(|&a, &b| {
        boxes[b]
            .confidence
            .partial_cmp(&boxes[a].confidence)
            .unwrap()
    });

    let mut result = vec![];

    while !idxs.is_empty() {
        // get the index with highest confidence
        let last_idx = 0;
        let last = idxs[last_idx];
        idxs.swap_remove(last_idx); // O(1) removal

        // add the picked box directly to the result
        result.push(boxes[last]);

        // remove overlapping boxes
        let last_box = &boxes[last];
        let mut i = 0;
        while i < idxs.len() {
            let current_box = &boxes[idxs[i]];
            if iou(last_box, current_box) > threshold {
                idxs.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_maximum_suppression() {
        let mut bboxes = vec![vec![
            BoundingBox {
                xmin: 0.0,
                ymin: 0.0,
                xmax: 1.0,
                ymax: 1.0,
                confidence: 0.5,
                class: 0,
            },
            BoundingBox {
                xmin: 0.0,
                ymin: 0.0,
                xmax: 1.0,
                ymax: 1.0,
                confidence: 0.5,
                class: 0,
            },
        ]];

        non_maximum_suppression(&mut bboxes, 0.5);

        assert_eq!(bboxes[0].len(), 1);
        assert_eq!(bboxes[0][0].confidence, 0.5);
    }

    #[test]
    fn test_non_maximum_suppression_fast() {
        let mut boxes = vec![
            BoundingBox {
                xmin: 0.0,
                ymin: 0.0,
                xmax: 1.0,
                ymax: 1.0,
                confidence: 0.5,
                class: 0,
            },
            BoundingBox {
                xmin: 0.0,
                ymin: 0.0,
                xmax: 1.0,
                ymax: 1.0,
                confidence: 0.5,
                class: 0,
            },
        ];

        let picked = non_maximum_suppression_fast(&mut boxes, 0.5);

        assert_eq!(picked.len(), 1);
        assert_eq!(picked[0].confidence, 0.5);
    }
}
