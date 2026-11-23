// Local pipe protocol. Request: "VIT1", uint32 frame ID, 3x224x224 LE floats.
// Reply: LE uint32 JSON byte count, JSON, then 1000 LE floats for each frame.
#pragma once
#include <cstdio>
#include <unistd.h>
#include <sys/resource.h>

using ProtocolPipe = std::unique_ptr<FILE, decltype(&std::fclose)>;

inline ProtocolPipe camera_output() {
    FILE* output = fdopen(dup(STDOUT_FILENO), "wb");
    if (!output || dup2(STDERR_FILENO, STDOUT_FILENO) < 0)
        throw std::runtime_error("Cannot separate camera protocol from MNN diagnostics");
    return ProtocolPipe(output, &std::fclose);
}

inline void camera_message(FILE* pipe, const std::string& message, const float* logits = nullptr) {
    uint32_t bytes = static_cast<uint32_t>(message.size());
    if (fwrite(&bytes, sizeof(bytes), 1, pipe) != 1 ||
        fwrite(message.data(), 1, bytes, pipe) != bytes ||
        (logits && fwrite(logits, sizeof(float), 1000, pipe) != 1000) || fflush(pipe))
        throw std::runtime_error("Camera reply pipe closed");
}

template <class Run>
void serve_camera(FILE* pipe, MNN::Tensor& input, MNN::Tensor& host, MNN::Tensor& output,
                  Run run, double load_ms, unsigned native_layers, bool profile) {
    using Clock = std::chrono::steady_clock;
    const auto ms = [](Clock::time_point t) {
        return std::chrono::duration<double,std::milli>(Clock::now() - t).count();
    };
    MNN::Tensor scores(&output, MNN::Tensor::CAFFE);
    if (host.elementSize() != 3*224*224 || scores.elementSize() != 1000 ||
        output.getType() != halide_type_of<float>() || !scores.host<float>())
        throw std::runtime_error("Camera mode expects 224x224 RGB and 1000 FP32 scores");
    camera_message(pipe, "{\"ready\":true,\"load_ms\":" + std::to_string(load_ms) + "}");
    for (;;) {
        char magic[4];
        auto n = fread(magic, 1, sizeof(magic), stdin);
        if (!n && feof(stdin)) return;
        uint32_t id;
        if (n != 4 || std::string(magic, 4) != "VIT1" || fread(&id,sizeof(id),1,stdin) != 1 ||
            fread(host.host<float>(),sizeof(float),host.elementSize(),stdin) != size_t(host.elementSize()))
            throw std::runtime_error("Incomplete or invalid camera tensor");
        const auto start = Clock::now();
        if (!input.copyFromHostTensor(&host)) throw std::runtime_error("Camera input copy failed");
        const double pack_ms = ms(start);
        mnn_fpga_profile_reset(profile);
        auto jobs_before = mnn_fpga_completed_jobs();
        auto ops_before = mnn_fpga_completed_int8_operations();
        const auto infer_start = Clock::now();
        run();
        const double infer_ms = ms(infer_start);
        auto jobs = mnn_fpga_completed_jobs() - jobs_before;
        auto ops = mnn_fpga_completed_int8_operations() - ops_before;
        if (ops != native_layers || (jobs > 0) != (native_layers > 0))
            throw std::runtime_error("Unexpected per-frame FPGA job or layer count");
        if (!output.copyToHostTensor(&scores)) throw std::runtime_error("Camera output copy failed");
        const double service = ms(start);
        rusage resources{};
        if (getrusage(RUSAGE_SELF, &resources)) throw std::runtime_error("Cannot read runner RSS");
        std::ostringstream meta;
        meta << "{\"id\":" << id << ",\"pack_ms\":" << pack_ms << ",\"infer_ms\":" << infer_ms
             << ",\"service_ms\":" << service << ",\"jobs\":" << jobs << ",\"layers\":" << ops
             << ",\"runner_peak_kib\":" << resources.ru_maxrss;
        if (native_layers) meta << ",\"native\":" << mnn_fpga_native_json();
        if (profile) meta << ",\"offload\":" << mnn_fpga_profile_json();
        meta << "}";
        camera_message(pipe, meta.str(), scores.host<float>());
    }
}
