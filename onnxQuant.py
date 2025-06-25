import argparse
import os
import math
import time
import cv2
import torch
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType 

class GetCropMatrix():
    def __init__(self, image_size, target_face_scale, align_corners=False):
        self.image_size = image_size
        self.target_face_scale = target_face_scale
        self.align_corners = align_corners

    def _compose_rotate_and_scale(self, angle, scale, shift_xy, from_center, to_center):
        cosv = math.cos(angle)
        sinv = math.sin(angle)
        fx, fy = from_center
        tx, ty = to_center
        acos = scale * cosv
        asin = scale * sinv

        a0 = acos
        a1 = -asin
        a2 = tx - acos * fx + asin * fy + shift_xy[0]

        b0 = asin
        b1 = acos
        b2 = ty - asin * fx - acos * fy + shift_xy[1]

        rot_scale_m = np.array([
            [a0, a1, a2],
            [b0, b1, b2],
            [0.0, 0.0, 1.0]
        ], np.float32)
        return rot_scale_m

    def process(self, scale, center_w, center_h):
        if self.align_corners:
            to_w, to_h = self.image_size - 1, self.image_size - 1
        else:
            to_w, to_h = self.image_size, self.image_size

        rot_mu = 0
        scale_mu = self.image_size / (scale * self.target_face_scale * 200.0)
        shift_xy_mu = (0, 0)
        matrix = self._compose_rotate_and_scale(
            rot_mu, scale_mu, shift_xy_mu,
            from_center=[center_w, center_h],
            to_center=[to_w / 2.0, to_h / 2.0])
        return matrix

class TransformPerspective():
    def __init__(self, image_size):
        self.image_size = image_size
        
    def process(self, image, matrix):
        return cv2.warpPerspective(
            image, matrix, dsize=(self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR, borderValue=0)

class Alignment:
    def __init__(self, input_size=256, target_face_scale=1.0, align_corners=True):
        self.input_size = input_size
        self.target_face_scale = target_face_scale
        self.align_corners = align_corners

        self.getCropMatrix = GetCropMatrix(image_size=self.input_size,
                                           target_face_scale=self.target_face_scale,
                                           align_corners=self.align_corners)
        self.transformPerspective = TransformPerspective(image_size=self.input_size)

    def preprocess(self, image, landmarks, scale, center_w, center_h):
        matrix = self.getCropMatrix.process(scale, center_w, center_h)
        input_tensor = self.transformPerspective.process(image, matrix)  # HxWxC BGR uint8
        input_tensor = input_tensor[np.newaxis, :]  # Add batch dim: 1xHxWxC
        input_tensor = torch.from_numpy(input_tensor).float().permute(0, 3, 1, 2)  # NCHW
        input_tensor = input_tensor / 255.0 * 2.0 - 1.0  # Normalize to [-1,1]
        return input_tensor, matrix

# Calibration DataReader using real images 

class RealDataCalibrationReader(CalibrationDataReader):
    def __init__(self, image_dir, metadata_path, alignment, max_samples=100):
        self.image_dir = image_dir
        self.metadata_path = metadata_path
        self.alignment = alignment
        self.max_samples = max_samples
        
        self.preprocessed_tensors = []
        self._prepare()
        self.enum_data_dicts = None

    def _prepare(self):
        count = 0
        with open(self.metadata_path, 'r') as f:
            for line in f:
                if count >= self.max_samples:
                    break
                item = line.strip().split("\t")
                if len(item) < 6:
                    continue
                image_name, landmarks_5pts_str, landmarks_gt, scale_str, center_w_str, center_h_str = item[:6]
                image_path = os.path.join(self.image_dir, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Failed to load {image_path}, skipping...")
                    continue

                landmarks_5pts = np.array(list(map(float, landmarks_5pts_str.split(","))), dtype=np.float32).reshape(5, 2)
                scale, center_w, center_h = float(scale_str), float(center_w_str), float(center_h_str)

                input_tensor, _ = self.alignment.preprocess(image, landmarks_5pts, scale, center_w, center_h)
                self.preprocessed_tensors.append(input_tensor.numpy())
                count += 1
        print(f"Loaded {len(self.preprocessed_tensors)} calibration samples.")

    def get_next(self):
        if self.enum_data_dicts is None:
            self.enum_data_dicts = iter([{"input": x.astype(np.float32)} for x in self.preprocessed_tensors])
        return next(self.enum_data_dicts, None)

# Main quantization code 

def main():
    parser = argparse.ArgumentParser(description="Post-Training Quantization")
    parser.add_argument("--modelpath", type=str, required=True, help="Path to FP32 ONNX model")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata .tsv file")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing calibration images")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Set working dir to script location

    model_fp32 = args.modelpath
    metadata = args.metadata
    img_dir = args.img_dir
    model_quant = os.path.splitext(model_fp32)[0] + "_quantized.onnx"

    alignment = Alignment()
    reader = RealDataCalibrationReader(img_dir, metadata, alignment)

    quantize_static(
        model_input=model_fp32,
        model_output=model_quant,
        calibration_data_reader=reader,
        quant_format=QuantType.QInt8
    )
    print(f"Quantization complete. Saved to: {model_quant}")

if __name__ == "__main__":
    main()