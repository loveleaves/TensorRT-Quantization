
// tensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
// 上一节的代码

class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override
    {
        if (severity <= Severity::kINFO)
        {
            printf("%d: %s\n", severity, msg);
        }
    }
} logger;

nvinfer1::Weights make_weights(float *ptr, int n)
{
    nvinfer1::Weights w;
    w.count = n;
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = ptr;
    return w;
}
bool build_model()
{
    TRTLogger logger;

    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);

    const int num_input = 3;
    const int num_output = 2;
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};
    float layer1_bias_values[] = {0.3, 0.8};

    // 创建输入 Tensor（Dims2）
    nvinfer1::ITensor *input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{1, num_input});

    // 创建权重和偏置
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, num_output * num_input);
    nvinfer1::Weights layer1_bias = make_weights(layer1_bias_values, num_output);

    // 权重 tensor（Dims2）
    nvinfer1::Dims weightDims{2, {num_output, num_input}};
    auto weightConstant = network->addConstant(weightDims, layer1_weight);
    nvinfer1::ITensor *weightTensor = weightConstant->getOutput(0);

    // 执行矩阵乘法
    auto matmul = network->addMatrixMultiply(*input, nvinfer1::MatrixOperation::kNONE,
                                             *weightTensor, nvinfer1::MatrixOperation::kTRANSPOSE);

    // 偏置 tensor（Dims2）
    nvinfer1::Dims biasDims{2, {1, num_output}};
    auto biasConstant = network->addConstant(biasDims, layer1_bias);
    nvinfer1::ITensor *biasTensor = biasConstant->getOutput(0);

    // 添加偏置
    auto fc_output = network->addElementWise(*matmul->getOutput(0), *biasTensor,
                                             nvinfer1::ElementWiseOperation::kSUM);

    // 激活层
    auto prob = network->addActivation(*fc_output->getOutput(0), nvinfer1::ActivationType::kSIGMOID);
    prob->getOutput(0)->setName("prob");

    // 标记输出
    network->markOutput(*prob->getOutput(0));

    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 28);

    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine)
    {
        printf("Build engine failed.\n");
        return false;
    }
    nvinfer1::IHostMemory *model_data = engine->serialize();
    FILE *f = fopen("engine.trtmodel", "wb");
    if (f)
    {
        fwrite(model_data->data(), 1, model_data->size(), f);
        fclose(f);
    }
    delete model_data;
    delete engine;
    delete network;
    delete config;
    delete builder;
    printf("Done.\n");
    return true;
}

vector<unsigned char> load_file(const string &file)
{
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0)
    {
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char *)&data[0], length);
    }
    in.close();
    return data;
}

void inference()
{

    // ------------------------------ 1. 准备模型并加载   ----------------------------
    TRTLogger logger;
    auto engine_data = load_file("engine.trtmodel");
    // 执行推理前，需要创建一个推理的runtime接口实例。与builer一样，runtime需要logger：
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    // 将模型从读取到engine_data中，则可以对其进行反序列化以获得engine
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (engine == nullptr)
    {
        printf("Deserialize cuda engine failed.\n");
        delete runtime;
        return;
    }

    nvinfer1::IExecutionContext *execution_context = engine->createExecutionContext();
    cudaStream_t stream = nullptr;
    // 创建CUDA流，以确定这个batch的推理是独立的
    cudaStreamCreate(&stream);

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

    // ------------------------------ 2. 准备好要推理的数据并搬运到GPU   ----------------------------
    float input_data_host[] = {1, 2, 3};
    float *input_data_device = nullptr;

    float output_data_host[2];
    float *output_data_device = nullptr;

    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);

    // 设置 GPU 绑定指针（只有 enqueueV2 需要，enqueueV3 不再需要此参数）
    execution_context->setTensorAddress("image", input_data_device);
    execution_context->setTensorAddress("prob", output_data_device);

    // 使用 enqueueV3（只传入 stream）
    bool success = execution_context->enqueueV3(stream);

    // 将结果从 GPU 复制回 CPU
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    printf("output_data_host = %f, %f\n", output_data_host[0], output_data_host[1]);

    // ------------------------------ 4. 释放内存 ----------------------------
    printf("Clean memory\n");
    cudaStreamDestroy(stream);
    delete execution_context;
    delete engine;
    delete runtime;

    // ------------------------------ 5. 手动推理进行验证 ----------------------------
    const int num_input = 3;
    const int num_output = 2;
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};
    float layer1_bias_values[] = {0.3, 0.8};

    printf("手动验证计算结果：\n");
    for (int io = 0; io < num_output; ++io)
    {
        float output_host = layer1_bias_values[io];
        for (int ii = 0; ii < num_input; ++ii)
        {
            output_host += layer1_weight_values[io * num_input + ii] * input_data_host[ii];
        }

        // sigmoid
        float prob = 1 / (1 + exp(-output_host));
        printf("output_prob[%d] = %f\n", io, prob);
    }
}

int main()
{

    if (!build_model())
    {
        return -1;
    }
    inference();
    return 0;
}