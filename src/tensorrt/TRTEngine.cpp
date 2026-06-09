#include "TRTEngine.h"
#include <iostream>

#if defined(MOCK_HIP)
class TRTEngine::Impl {
public:
    void load_model(const std::string& filepath) {
        std::cout << "[TRTEngine] Mock loading TensorRT plan: " << filepath << std::endl;
    }
    void launch(void* input_ptr, void* output_ptr, int seq_len) {
        // Mock launch execution
        std::cout << "[TRTEngine] Mock launching TensorRT engine" << std::endl;
    }
};
#else
// Ensure proper TensorRT integration for the NVIDIA compute layer routing mandate
#include <NvInfer.h>
#include <fstream>

class TRTEngineLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

class TRTEngine::Impl {
public:
    Impl() : runtime_(nullptr), engine_(nullptr), context_(nullptr) {}

    ~Impl() {
        if (context_) context_->destroy();
        if (engine_) engine_->destroy();
        if (runtime_) runtime_->destroy();
    }

    void load_model(const std::string& filepath) {
        std::cout << "[TRTEngine] Loading TensorRT plan: " << filepath << std::endl;
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        if (!file.good()) {
            throw std::runtime_error("Failed to read TensorRT plan file.");
        }

        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> trtModelStream(size);
        file.read(trtModelStream.data(), size);

        runtime_ = nvinfer1::createInferRuntime(gLogger);
        if (!runtime_) throw std::runtime_error("Failed to create TensorRT InferRuntime.");

        engine_ = runtime_->deserializeCudaEngine(trtModelStream.data(), size);
        if (!engine_) throw std::runtime_error("Failed to deserialize TensorRT CUDA engine.");

        context_ = engine_->createExecutionContext();
        if (!context_) throw std::runtime_error("Failed to create TensorRT ExecutionContext.");
    }

    void launch(void* input_ptr, void* output_ptr, int seq_len) {
        if (!context_) {
            throw std::runtime_error("TensorRT execution context is not initialized.");
        }

        // TensorRT Native Execution using enqueueV3 for the latest API standards
        // Note: Buffer addressing must be mapped identically to the bindings schema
        context_->setInputTensorAddress("input", input_ptr);
        context_->setTensorAddress("output", output_ptr);

        // Ensure execution triggers seamlessly against the default stream
        if (!context_->enqueueV3(0)) {
            throw std::runtime_error("TensorRT enqueueV3 execution failed.");
        }
    }

private:
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
};
#endif

TRTEngine::TRTEngine() : impl_(std::make_unique<Impl>()) {}
TRTEngine::~TRTEngine() = default;

void TRTEngine::load_model(const std::string& filepath) {
    impl_->load_model(filepath);
}

void TRTEngine::launch(void* input_ptr, void* output_ptr, int seq_len) {
    impl_->launch(input_ptr, output_ptr, seq_len);
}
