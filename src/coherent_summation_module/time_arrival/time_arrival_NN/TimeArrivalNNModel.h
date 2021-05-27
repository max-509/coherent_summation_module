#ifndef _TIME_ARRIVAL_NN_MODEL_H
#define _TIME_ARRIVAL_NN_MODEL_H

#include "TimeArrivalNNBase.h"

#include <vector>
#include <memory>
#include <string>
#include <tuple>
#include <cstddef>
#include <functional>

//saved_model_cli show --dir "model_dir" --tag_set serve --signature_def serving_default

//In Python3
/*
loaded = tf.saved_model.load('layered_ANN_model')
print(list(loaded.signatures.keys()))  # ["signature_def"]

infer = loaded.signatures["signature_def"]
*/

//saved_model_cli show --dir "model_dir" --all

class TimeArrivalWrapperNN;

class TimeArrivalNNModel : public TimeArrivalNNBase {
public:
    TimeArrivalNNModel(const std::string &model_path,
                       char const *const *tags,
                       int ntags,
                       std::vector<std::pair<std::string, int>> &input_ops,
                       std::vector<std::pair<std::string, int>> &output_ops) noexcept(false);

    std::vector<float> process(std::vector<float> &, std::vector<float> &, std::vector<float> &) noexcept(false) final;

    float process(float, float, float) noexcept(false) final;

    ~TimeArrivalNNModel() override = default;

private:

    TF_Buffer *run_opts_ = nullptr;
    std::shared_ptr<TF_SessionOptions> sess_opts_;
    std::string tags_str_;
    char const *tags_;
    int ntags_;
    std::string model_path_;
    std::shared_ptr<TF_Session> session_;
    std::vector<TF_Output> outputs_;
    std::vector<TF_Output> inputs_;

};

#endif //_TIME_ARRIVAL_NN_MODEL_H
