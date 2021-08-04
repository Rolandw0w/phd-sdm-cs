SET(CMAKE_CUDA_COMPILER /usr/local/cuda-11.1/bin/nvcc)

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
        core/runners/cs2_runner.cuh
        core/runners/cs2_runner.cu
        core/cuda/functions.cuh
        core/sdm/sdm_cs1.cuh
        core/sdm/sdm_cs2.cuh
        core/sdm/sdm_kanerva.cuh
        core/runners/kanerva_runner.cuh
        core/runners/kanerva_runner.cu
        core/sdm/sdm_cs2_s2.cuh core/runners/cs2_s2_runner.cu core/runners/cs2_s2_runner.cuh core/runners/synth_runner.cu core/runners/synth_runner.cuh)

set_property(TARGET sdm_cs PROPERTY CUDA_ARCHITECTURES 72)

set_source_files_properties(core/data_ops/data_reader.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(core/data_ops/data_writer.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(core/data_ops/image_writer.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(core/utils/utils.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(
        sdm_cs
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
