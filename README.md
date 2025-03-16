# TensorRT-Quantization
using TensorRT to quantize model and infer in GPU

## References
- [tensorrt docs](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/quick-start-guide.html)
- [tensorrt api](https://docs.nvidia.com/deeplearning/tensorrt/archives/)
- [install tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/installing.html)
- [intro-quantization](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1070/developer-guide/index.html#intro-quantization)
- [TensorRT samples](https://github.com/NVIDIA/TensorRT/tree/main/samples)
- [Implementation of popular deep learning networks with TensorRT](https://github.com/wang-xinyu/tensorrtx)

## introduction to quantization


## Post-Training Quantization Using Calibration, PTQ
- Calibration is only applicable to INT8 quantization. [tensorrt-1007](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1070/developer-guide/index.html#enable_int8_c)

In post-training quantization, TensorRT computes a scale value for each tensor in the network. This process, called calibration, requires you to supply representative input data on which TensorRT runs the network to collect statistics for each activation tensor.

### calibrator
在训练后量化过程中，TensorRT需要计算模型中每个张量的比例因子，这个过程被称为校准。校准过程中需要提供具有代表性的数据，以便TensorRT在这个数据集上运行模型然后收集每个张量的统计信息用于寻找一个最佳的比例因子。寻找最佳比例因子需要平衡离散化误差（随着每个量化值表示的范围变大而变大）和截断误差（其值被限制在可表示范围的极限内）这两个误差源，TensorRT提供了几种不同的校准器：
- IInt8EntropyCalibrator2：当前推荐的熵校准器，默认情况下校准发生在层融合之前，推荐用于CNN模型中。
- IInt8MinMaxCalibrator：该校准器使用激活分布的整个范围来确定比例因子，默认情况下校准发生在层融合之前，推荐用于NLP任务的模型中。
- IInt8EntropyCalibrator： 该校准器是TensorRT最原始的熵校准器，默认情况下校准发生在层融合之后，目前已不推荐使用。
- IInt8LegacyCalibrator： 该校准器需要用户进行参数化，默认情况下校准发生在层融合之后，不推荐使用。

TensorRT构建INT8模型引擎时，会执行下面的步骤：

1. 构建一个32位的模型引擎，然后在校准数据集上运行这个引擎，然后为每个张量激活值的分布记录一个直方图；
2. 从直方图构建一个校准表，为每个张量计算出一个比例因子；
3. 根据校准表和模型的定义构建一个INT8的引擎。

校准的过程可能会比较慢，不过第二步生成的校准表可以输出到文件并可以被重用，如果校准表文件已存在，那么校准器就直接从该文件中读取校准表而无需执行前面两步。另外，与引擎文件不同的是，校准表是可以跨平台使用的。因此，我们在实际部署模型过程中可以先在带通用GPU的计算机上生成校准表，然后在Jetson Nano等嵌入式平台上去使用。为了编码方便，我们可以用Python编程来实现INT8量化过程来生成校准表。

## TensorRT推理部署方案
- [learning-cuda-trt code](https://github.com/jinmin527/learning-cuda-trt)

### C++硬代码方案
- 代表：[tensorrtx](https://github.com/wang-xinyu/tensorrtx)
- 流程：C++硬代码=》TRT API =》 TRT Builder =》 TRT Engine

### ONNX方案
- 流程：ONNX(libnvonnxparser.so) =》TRT API =》 TRT Builder =》 TRT Engine
- 一般思路：
    1. 导出模型onnx，查看输入和输出。
    2. 查看代码，找到onnx的预处理，分析预处理逻辑
    3. 利用上述信息实现onnx py推理实现
    4. 验证正常可实现C++推理实现

- 其他：
  - 1. 模型优化：
    - 导出前：模型蒸馏、剪枝、量化等
    - 导出时及之后：算子融合
  - 2. 推理优化：图融合、图调度等
- 工具：
  - 优化：onnx-simplifier，简化 ONNX 计算图 的 Python 工具，能够移除冗余算子、融合计算节点
  - 模型结构：netron，在线或离线查看模型结构


### TensorRT库文件
- libnvinfer.so：TensorRT核心库
- libnvinfer_plugin.so：nvidia官方提供的插件，[github](https://github.com/NVIDIA/TensorRT/tree/main/plugin)
- libprotobuf.so：protobuf库
- libnvonnxparser.so：ONNX解析

## object detection
### YOLO
- [YOLOv8](./object_detection/YOLOv8/README.md)

## TensorRT-LLM
tensorRT llm backend
- [TensorRT-LLM](./TensorRT-LLM/README.md)

## deploy
deploy tensorRT engine
- [deploy using triton-server](./deploy/README.md)

## other
- [Int8 maybe slower than fp16](https://github.com/NVIDIA/TensorRT/issues/993)