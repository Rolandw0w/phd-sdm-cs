cmake_minimum_required(VERSION 3.17)
project(sdm_cs CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(
        sdm_cs core/main.cu
        core/runners/base_runner.cuh
        core/runners/cifar10_runner.cu
        core/runners/cifar10_runner.cuh
        core/data_ops/data_reader.cpp
        core/data_ops/data_reader.hpp
        core/data_ops/data_writer.cpp
        core/data_ops/data_writer.hpp
        core/data_ops/image_writer.cpp
        core/data_ops/image_writer.hpp
        core/cuda/kernels.cuh
        core/runners/labels_runner.cu
        core/runners/labels_runner.cuh
        core/main.cu
        core/main.cuh
        core/sdm/sdm_base.cuh
        core/sdm/sdm_jaeckel.cuh
        core/sdm/sdm_labels.cuh
        core/utils/utils.cpp
        core/utils/utils.hpp
        core/runners/cs1_runner.cuh
        core/runners/cs1_runner.cu
        core/cuda/functions.cuh core/sdm/sdm_cs1.cuh core/sdm/sdm_kanerva.cuh core/runners/kanerva_runner.cuh core/runners/kanerva_runner.cu)

set_source_files_properties(core/data_ops/data_reader.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(core/data_ops/data_writer.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(core/data_ops/image_writer.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(core/utils/utils.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(
        sdm_cs
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
