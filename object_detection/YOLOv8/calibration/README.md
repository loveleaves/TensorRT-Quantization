# calibration
## Post-Training Quantization Using Calibration, PTQ
- Calibration is only applicable to INT8 quantization. [tensorrt-1007](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1070/developer-guide/index.html#enable_int8_c)

In post-training quantization, TensorRT computes a scale value for each tensor in the network. This process, called calibration, requires you to supply representative input data on which TensorRT runs the network to collect statistics for each activation tensor.

## Quantization-aware-training, QAT
- [量化感知训练（Quantization-aware-training）探索-从原理到实践](https://zhuanlan.zhihu.com/p/548174416)