#include "calibrator.h"
#include "filesystem.hpp"

#include <stdio.h>
#include <math.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <chrono>
#include <assert.h>

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

namespace fs = ghc::filesystem;

// 以下示例捕获所有警告消息，但忽略信息性消息
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // 抑制信息级别的消息
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s [engine_path] [image_dir]\n", argv[0]);
        return -1;
    }
    const char *engine_file_path = argv[1];
    const char *int8CalibTable = "int8calib.table";
    const fs::path path{argv[2]};

    if (!fs::is_directory(path))
    {
        std::cout << "wrong calibration folder!" << std::endl;
        std::abort();
    }

    // 实例化ILogger
    Logger logger;

    // 创建builder
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));

    // 创建网络(显性batch)
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));

    // 创建ONNX解析器：parser
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    // 读取文件
    parser->parseFromFile(engine_file_path, static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));

    // 创建构建配置，用来指定trt如何优化模型
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    // 设定配置
    // 工作空间大小
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);
    // 设置精度
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    auto *calibrator = new Int8Calibrator(1, 640, 640, argv[2], int8CalibTable, "");
    config->setInt8Calibrator(calibrator);

    // 创建引擎
    auto engine = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));

    // 序列化保存engine
    std::ofstream engine_file(engine_file_path, std::ios::binary);
    assert(engine_file.is_open() && "Failed to open engine file");
    engine_file.write((char *)engine->data(), engine->size());
    engine_file.close();

    std::cout << "Engine build success!" << std::endl;
    return 0;
}