cmake_minimum_required(VERSION 3.16)

project(anonymous_artifact LANGUAGES CXX)

# Use C++17 for all experiments.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Put compiled executables into build/bin.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Header-only library target for shared implementation.
# All .hpp/.h files are stored in include/.
add_library(artifact_headers INTERFACE)
target_include_directories(artifact_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Optional OpenMP support.
# If your code uses OpenMP, CMake will link it automatically when available.
# If your code does not use OpenMP, this part will not affect normal compilation.
find_package(OpenMP QUIET)

function(add_experiment target_name source_file)
    add_executable(${target_name} ${source_file})

    target_link_libraries(${target_name} PRIVATE artifact_headers)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(MSVC)
        target_compile_options(${target_name} PRIVATE /O2)
    else()
        target_compile_options(${target_name} PRIVATE -O3)
    endif()
endfunction()

# Main experiments
add_experiment(run_sift1m_overall
    ${CMAKE_CURRENT_SOURCE_DIR}/experiments/sift1m_overall_main.cpp
)

add_experiment(run_ciaodvd_overall
    ${CMAKE_CURRENT_SOURCE_DIR}/experiments/CiaoDVD_overall_main.cpp
)

add_experiment(run_kuairand_industrial_validation
    ${CMAKE_CURRENT_SOURCE_DIR}/experiments/KuaiRand_industrial_validation.cpp
)

# Ablation and development experiments
add_experiment(run_full_two_stage_fusion
    ${CMAKE_CURRENT_SOURCE_DIR}/experiments/Full_two_Stage_fusion.cpp
)

add_experiment(run_stage1_only
    ${CMAKE_CURRENT_SOURCE_DIR}/experiments/Stage1_only.cpp
)

add_experiment(run_stage2_basic_1v1
    ${CMAKE_CURRENT_SOURCE_DIR}/experiments/Stage2_Basic_1v1.cpp
)

add_experiment(run_stage2_1v1_oversampling
    ${CMAKE_CURRENT_SOURCE_DIR}/experiments/Stage2_1v1_oversampling.cpp
)

add_experiment(run_stage2_2v2_oversampling
    ${CMAKE_CURRENT_SOURCE_DIR}/experiments/Stage2_2v2_oversampling.cpp
)