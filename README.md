# TensorRT-Quantization
using TensorRT to quantize model and infer in GPU

## References
- [tensorrt docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)
- [install tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)
- [intro-quantization](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1070/developer-guide/index.html#intro-quantization)
- [TensorRT samples](https://github.com/NVIDIA/TensorRT/tree/main/samples)
- [Implementation of popular deep learning networks with TensorRT](https://github.com/wang-xinyu/tensorrtx)

## introduction to quantization


## Post-Training Quantization Using Calibration, PTQ
- Calibration is only applicable to INT8 quantization. [tensorrt-1007](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1070/developer-guide/index.html#enable_int8_c)

In post-training quantization, TensorRT computes a scale value for each tensor in the network. This process, called calibration, requires you to supply representative input data on which TensorRT runs the network to collect statistics for each activation tensor.

## object detection
### YOLO
- [YOLOv8](./object_detection/YOLOv8/README.md)

## other
- [Int8 maybe slower than fp16](https://github.com/NVIDIA/TensorRT/issues/993)