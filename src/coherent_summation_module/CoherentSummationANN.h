#ifndef COHERENT_SUMMATION_ANN_H
#define COHERENT_SUMMATION_ANN_H

#include "py_common.h"

#include <memory>
#include <cstddef>
#include <string>
#include <vector>
#include <tuple>

#include "CoherentSummation.h"
#include "TimeArrivalWrapperNN.h"
enum NN_Type {
    FROZEN,
    MODEL
};

struct splitting_by_coord {
    double coord_0;
    double coord_1;
    std::size_t coord_dim;
};

class CoherentSummationANN final : public CoherentSummation {

public:

    CoherentSummationANN(const std::string &path_NN, NN_Type nn_type,
                         std::vector<std::pair<std::string, int>> &input_ops,
                         std::vector<std::pair<std::string, int>> &output_ops,
                         splitting_by_coord x_splitting,
                         splitting_by_coord z_splitting,
                         splitting_by_coord y_splitting);

    CoherentSummationANN(const std::string &path_NN, NN_Type nn_type,
                         std::vector<std::pair<std::string, int>> &input_ops,
                         std::vector<std::pair<std::string, int>> &output_ops,
                         py_array_d environment);

    py_array_d
    emission_tomography_method(py_array_d gather,
                               py_array_d receivers_coords,
                               double dt,
                               py_array_d tensor_matrix,
                               std::ptrdiff_t receivers_block_size = 20,
                               std::ptrdiff_t samples_block_size = 1000);

    py_array_d
    emission_tomography_method(py_array_d gather,
                               double dt,
                               py_array_d receivers_coords,
                               std::ptrdiff_t receivers_block_size = 20,
                               std::ptrdiff_t samples_block_size = 1000);

    ~CoherentSummationANN() noexcept override;

private:
    py_array_d environment_{};
    TimeArrivalWrapperNN time_arrival_nn_;
private:
    void generate_environment_grid(splitting_by_coord x_splitting,
                                   splitting_by_coord z_splitting,
                                   splitting_by_coord y_splitting);

    void copy_environment(py_array_d environment);

    void load_ann(const std::string &path_NN, NN_Type nn_type,
                  std::vector<std::pair<std::string, int>> &input_ops,
                  std::vector<std::pair<std::string, int>> &output_ops);

    py_array_f prepare_times_table(py_array_d receivers_coords);
};


#endif //COHERENT_SUMMATION_ANN_H
