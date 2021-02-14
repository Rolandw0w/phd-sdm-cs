cmake_minimum_required(VERSION 3.17)
project(sdm_cs CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(
        sdm_cs main.cu
        runners/base_runner.cuh
        runners/cifar10_runner.cu
        runners/cifar10_runner.cuh
        data_ops/data_reader.cpp
        data_ops/data_reader.h
        data_ops/data_writer.cpp
        data_ops/data_writer.h
        data_ops/image_writer.cpp
        data_ops/image_writer.h
        cuda/kernels.cuh
        runners/labels_runner.cu
        runners/labels_runner.cuh
        main.cu
        main.cuh
        sdm/sdm_base.cuh
        sdm/sdm_jaeckel.cuh
        sdm/sdm_labels.cuh
        utils/utils.cpp
        utils/utils.h
)
set_source_files_properties(data_ops/data_reader.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(data_ops/data_writer.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(data_ops/image_writer.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(utils/utils.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(
        sdm_cs
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

