cmake_minimum_required(VERSION 3.17)
project(sdm_cs CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(
        sdm_cs main.cu
        base_runner.cuh
        cifar10_runner.cu
        cifar10_runner.cuh
        data_reader.cpp
        data_reader.h
        data_writer.cpp
        data_writer.h
        image_writer.cpp
        image_writer.h
        kernels.cuh
        labels_runner.cu
        labels_runner.cuh
        main.cu
        main.cuh
        sdm_base.cuh
        sdm_jaeckel.cuh
        sdm_labels.cuh
        utils.cpp
        utils.h
)
set_source_files_properties(data_reader.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(data_writer.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(image_writer.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(utils.cpp PROPERTIES LANGUAGE CUDA)

set_target_properties(
        sdm_cs
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

