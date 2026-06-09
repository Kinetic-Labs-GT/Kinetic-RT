#pragma once

#include <string>
#include <vector>
#include <memory>

class TRTEngine {
public:
    TRTEngine();
    ~TRTEngine();

    void load_model(const std::string& filepath);
    void launch(void* input_ptr, void* output_ptr, int seq_len);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
