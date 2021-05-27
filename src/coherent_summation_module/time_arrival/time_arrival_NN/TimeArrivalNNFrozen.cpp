#include "TimeArrivalNNFrozen.h"
#include "TimeArrivalNNException.h"

#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <tuple>
#include <functional>
#include <memory>

TimeArrivalNNFrozen::TimeArrivalNNFrozen(const std::string &pb_filename,
                                         std::vector<std::pair<std::string, int>> &input_ops,
                                         std::vector<std::pair<std::string, int>> &output_ops) noexcept(false) :
        graph_opts_(TF_NewImportGraphDefOptions(), TF_DeleteImportGraphDefOptions) {
    TF_Buffer *graph_def = read_pb_file(pb_filename);

    TF_GraphImportGraphDef(graph_.get(), graph_def, graph_opts_.get(), status_.get());

    if (TF_GetCode(status_.get()) != TF_OK) {
        throw TimeArrivalNNException(std::string("ERROR: Unable to import graph: ") + TF_Message(status_.get()));
    }

    session_opts_ = std::shared_ptr<TF_SessionOptions>(TF_NewSessionOptions(), TF_DeleteSessionOptions);
    uint8_t config[7] = {0x32, 0x5, 0x20, 0x1, 0x2a, 0x01, 0x30}; // protobuf data for auto memory gpu_options.allow_growth=True and gpu_options.visible_device_list="0"
    TF_SetConfig(session_opts_.get(), static_cast<void*>(config), 7, status_.get());
    if (TF_GetCode(status_.get()) != TF_OK) {
        throw TimeArrivalNNException(std::string("ERROR: Unable to set options: ") + TF_Message(status_.get()));
    }
    session_ = std::shared_ptr<TF_Session>(
            TF_NewSession(graph_.get(), session_opts_.get(), status_.get()),
            std::bind(TF_DeleteSession, std::placeholders::_1, status_.get()));

    if (TF_GetCode(status_.get()) != TF_OK) {
        throw TimeArrivalNNException(std::string("ERROR: Unable to open session: ") + TF_Message(status_.get()));
    }

    for (const auto &in_op : input_ops) {
        TF_Operation *input_op = TF_GraphOperationByName(graph_.get(), in_op.first.c_str());
        TF_Output input_op_input = {input_op, in_op.second};
        inputs_.push_back(input_op_input);
    }

    for (const auto &out_op : output_ops) {
        TF_Operation *output_op = TF_GraphOperationByName(graph_.get(), out_op.first.c_str());
        TF_Output output_op_output = {output_op, out_op.second};
        outputs_.push_back(output_op_output);
    }
}

float TimeArrivalNNFrozen::process(float x_p, float z_p, float x_r) noexcept(false) {
    static const auto empty_deallocator = [](void *data, std::size_t len, void *arg) {};
    static int64_t const coord_dims[] = {1, 1};
    static int64_t const time_dims[] = {1};
    TF_Tensor *x_p_tensor = TF_NewTensor(TF_FLOAT, coord_dims, 2, &x_p, sizeof(float), empty_deallocator, nullptr);
    TF_Tensor *z_p_tensor = TF_NewTensor(TF_FLOAT, coord_dims, 2, &z_p, sizeof(float), empty_deallocator, nullptr);
    TF_Tensor *x_r_tensor = TF_NewTensor(TF_FLOAT, coord_dims, 2, &x_r, sizeof(float), empty_deallocator, nullptr);

    TF_Tensor *const input_tensors[] = {x_p_tensor, z_p_tensor, x_r_tensor};

    TF_Tensor *time_tensor = TF_AllocateTensor(TF_FLOAT, time_dims, 1, sizeof(float));

    TF_SessionRun(session_.get(), nullptr,
                  inputs_.data(), input_tensors, inputs_.size(),
                  outputs_.data(), &time_tensor, outputs_.size(),
                  nullptr, 0, nullptr, status_.get());

    if (TF_GetCode(status_.get()) != TF_OK) {
        throw TimeArrivalNNException(std::string("ERROR: Unable to run session: ") + TF_Message(status_.get()));
    }

    float t = *static_cast<float *>(TF_TensorData(time_tensor));

    TF_DeleteTensor(time_tensor);

    return t;
}

std::vector<float>
TimeArrivalNNFrozen::process(std::vector<float> &x_p, std::vector<float> &z_p, std::vector<float> &x_r) noexcept(false) {
    const int64_t N = static_cast<int64_t>(std::min(std::min(x_p.size(), z_p.size()), x_r.size()));
    static const auto empty_deallocator = [](void *data, std::size_t len, void *arg) {};
    static int64_t const coord_dims[] = {N, 1};
    static int64_t const time_dims[] = {N};
    TF_Tensor *x_p_tensor = TF_NewTensor(TF_FLOAT, coord_dims, 2, (void *) (x_p.data()), N * sizeof(float),
                                         empty_deallocator, nullptr);
    TF_Tensor *z_p_tensor = TF_NewTensor(TF_FLOAT, coord_dims, 2, (void *) (z_p.data()), N * sizeof(float),
                                         empty_deallocator, nullptr);
    TF_Tensor *x_r_tensor = TF_NewTensor(TF_FLOAT, coord_dims, 2, (void *) (x_r.data()), N * sizeof(float),
                                         empty_deallocator, nullptr);

    TF_Tensor *const input_tensors[] = {x_p_tensor, z_p_tensor, x_r_tensor};

    TF_Tensor *time_tensor = TF_AllocateTensor(TF_FLOAT, time_dims, 1, N * sizeof(float));

    TF_SessionRun(session_.get(), nullptr,
                  inputs_.data(), input_tensors, inputs_.size(),
                  outputs_.data(), &time_tensor, outputs_.size(),
                  nullptr, 0, nullptr, status_.get());

    if (TF_GetCode(status_.get()) != TF_OK) {
        throw TimeArrivalNNException(std::string("ERROR: Unable to run session: ") + TF_Message(status_.get()));
    }

    auto *times = static_cast<float *>(TF_TensorData(time_tensor));

    std::vector<float> times_vec{times, times + N};

    TF_DeleteTensor(time_tensor);

    return times_vec;
}

TF_Buffer *TimeArrivalNNFrozen::read_pb_file(const std::string &pb_filename) {
    FILE *f;
    f = fopen(pb_filename.c_str(), "rb");
    if (f == nullptr) {
        return nullptr;
    }

    fseek(f, 0, SEEK_END);
    long int fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    void *data = malloc(fsize);
    if (data == nullptr) {
        return nullptr;
    }
    auto n_read = fread(data, fsize, 1, f);
    if (n_read < fsize) {
        fclose(f);
        return nullptr;
    }
    fclose(f);

    TF_Buffer *buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = [](void *data, std::size_t length) { free(data); };

    return buf;
}