# Models

These pre-trained models are redistributed unmodified under their original licenses.

| File | Purpose | Source | License |
| --- | --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | Face detection + 5 landmarks (YuNet) | [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT |
| `face_recognition_sface_2021dec.onnx` | Face embeddings (SFace) | [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | Apache 2.0 |
| `person_detection_mediapipe_2023mar.onnx` | Person detection for body pose (MediaPipe BlazePose) | [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/person_detection_mediapipe) | Apache 2.0 |
| `pose_estimation_mediapipe_2023mar.onnx` | 33 body landmarks (MediaPipe BlazePose) | [OpenCV Model Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/pose_estimation_mediapipe) | Apache 2.0 |
| `emotion-ferplus-8.onnx` | Facial expression (FER+) | [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus) | MIT |

`build.ps1` downloads any that are missing.
