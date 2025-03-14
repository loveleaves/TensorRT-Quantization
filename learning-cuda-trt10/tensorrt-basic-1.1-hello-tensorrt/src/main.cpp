
// tensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>

class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override
    {
        if (severity <= Severity::kVERBOSE)
        {
            printf("%d: %s\n", severity, msg);
        }
    }
};

nvinfer1::Weights make_weights(float *ptr, int n)
{
    nvinfer1::Weights w;
    w.count = n;
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = ptr;
    return w;
}

int main()
{
    // 本代码主要实现一个最简单的神经网络 figure/simple_fully_connected_net.png

    TRTLogger logger; // logger是必要的，用来捕捉warning和info等

    // ----------------------------- 1. 定义 builder, config 和network -----------------------------
    // 这是基本需要的组件
    // 形象的理解是你需要一个builder去build这个网络，网络自身有结构，这个结构可以有不同的配置
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    // 创建一个构建配置，指定TensorRT应该如何优化模型，tensorRT生成的模型只能在特定配置下运行
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    // 创建网络定义，其中createNetworkV2(1)表示采用显性batch size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
    // 因此贯穿以后，请都采用createNetworkV2(1)而非createNetworkV2(0)或者createNetwork
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);

    // 构建一个模型
    /*
        Network definition:

        image
          |
        linear (fully connected)  input = 3, output = 2, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5]], b=[0.3, 0.8]
          |
        sigmoid
          |
        prob
    */

    // ----------------------------- 2. 输入，模型结构和输出的基本信息 -----------------------------
    const int num_input = 3;                                       // in_channel
    const int num_output = 2;                                      // out_channel
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5}; // 前3个给w1的rgb，后3个给w2的rgb
    float layer1_bias_values[] = {0.3, 0.8};

    // 输入指定数据的名称、数据类型和完整维度，将输入层添加到网络
    nvinfer1::ITensor *input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2(1, num_input));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 6);
    nvinfer1::Weights layer1_bias = make_weights(layer1_bias_values, 2);
    // 添加全连接层

    // 权重 tensor（Dims2）
    nvinfer1::Dims weightDims{2, {num_output, num_input}};
    auto weightConstant = network->addConstant(weightDims, layer1_weight);
    nvinfer1::ITensor *weightTensor = weightConstant->getOutput(0);

    // **确保 input 是 2D**
    // auto reshape = network->addShuffle(*input);
    // reshape->setReshapeDimensions(nvinfer1::Dims2{1, num_input});
    // nvinfer1::ITensor *reshaped_input = reshape->getOutput(0);

    // 执行矩阵乘法
    auto matmul = network->addMatrixMultiply(*input, nvinfer1::MatrixOperation::kNONE,
                                             *weightTensor, nvinfer1::MatrixOperation::kTRANSPOSE);

    // 偏置 tensor（Dims1）
    nvinfer1::Dims biasDims{2, {1, num_output}};
    auto biasConstant = network->addConstant(biasDims, layer1_bias);
    nvinfer1::ITensor *biasTensor = biasConstant->getOutput(0);

    // **扩展 bias 维度，使其变成 [1, num_output]**
    // auto bias_reshape = network->addShuffle(*biasTensor);
    // bias_reshape->setReshapeDimensions(nvinfer1::Dims2{1, num_output});
    // nvinfer1::ITensor *reshaped_bias = bias_reshape->getOutput(0);

    // 添加偏置
    auto fc_output = network->addElementWise(*matmul->getOutput(0), *biasTensor,
                                             nvinfer1::ElementWiseOperation::kSUM);

    // 添加激活层
    auto prob = network->addActivation(*fc_output->getOutput(0), nvinfer1::ActivationType::kSIGMOID); // 注意更严谨的写法是*(layer1->getOutput(0)) 即对getOutput返回的指针进行解引用
    prob->getOutput(0)->setName("prob");

    // 将我们需要的prob标记为输出
    network->markOutput(*prob->getOutput(0));

    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f); // 256Mib
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 28);

    // ----------------------------- 3. 生成engine模型文件 -----------------------------
    // TensorRT 7.1.0版本已弃用buildCudaEngine方法，统一使用buildEngineWithConfig方法
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr)
    {
        printf("Build engine failed.\n");
        return -1;
    }

    // ----------------------------- 4. 序列化模型文件并存储 -----------------------------
    // 将模型序列化，并储存为文件
    nvinfer1::IHostMemory *model_data = engine->serialize();
    FILE *f = fopen("engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    delete model_data;
    delete engine;
    delete network;
    delete config;
    delete builder;
    printf("Done.\n");
    return 0;
}