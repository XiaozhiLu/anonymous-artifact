# Anonymous Artifact

This repository contains the C++ implementation and preprocessing scripts for the submitted paper. The code is organized to support reproduction of the main experiments, ablation studies, and industrial validation experiments reported in the paper.

The implementation consists of shared C++ header files and multiple experiment-specific `main.cpp` entry files. The shared algorithmic components are placed under `include/`, while each file under `experiments/` corresponds to one experiment or one group of experiments in the paper.

## Repository Structure

```text
anonymous_artifact/
├── CMakeLists.txt
├── README.md
│
├── include/
│   ├── binary_io.hpp
│   ├── group_task_eval.h
│   ├── retrieval_framework.hpp
│   ├── stimer.hpp
│   └── vector_ops.hpp
│
├── experiments/
│   ├── sift1m_overall_main.cpp
│   ├── CiaoDVD_overall_main.cpp
│   ├── KuaiRand_industrial_validation.cpp
│   ├── Full_two_Stage_fusion.cpp
│   ├── Stage1_only.cpp
│   ├── Stage2_Basic_1v1.cpp
│   ├── Stage2_1v1_oversampling.cpp
│   └── Stage2_2v2_oversampling.cpp
│
├── scripts/
│   ├── prepare_kuairand1k_itemgraph_ctr.py
│   └── preprocess_ciaodvd.py
│
└── data/
    └── README.md
```

## Requirements

### C++ Requirements

- CMake >= 3.16
- C++17-compatible compiler
- GCC, Clang, or Microsoft Visual Studio C++ compiler

### Python Requirements

The preprocessing scripts require Python 3.8 or later and the following packages:

```text
numpy
pandas
torch
```

If needed, install them with:

```bash
pip install numpy pandas torch
```

## Data Preparation

The raw datasets are not included in this repository due to file size and dataset license considerations.

Please follow the instructions in:

```text
data/README.md
```

to download and preprocess the datasets.

The datasets used in the paper include:

- SIFT1M
- CiaoDVD
- KuaiRand-1K

For KuaiRand-1K, the preprocessing script is:

```text
scripts/prepare_kuairand1k_itemgraph_ctr.py
```

For CiaoDVD, the preprocessing script is:

```text
scripts/preprocess_ciaodvd.py
```

After preprocessing, the generated files should be placed under the corresponding subdirectories of:

```text
data/processed/
```

Please make sure that the paths used in the C++ experiment files are consistent with the processed data directory.

## Build Instructions

From the root directory of this repository, run:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

After successful compilation, the executable files will be generated under the build directory. Depending on the operating system and compiler, they may appear in one of the following directories:

```text
build/bin/
build/bin/Release/
```

## Running Experiments

Each experiment is compiled as an independent executable. The mapping between paper experiments, source files, and executable names is shown below.

| Paper Experiment | Source File | Executable |
|---|---|---|
| Overall performance on SIFT1M | `experiments/sift1m_overall_main.cpp` | `run_sift1m_overall` |
| Overall performance on CiaoDVD | `experiments/CiaoDVD_overall_main.cpp` | `run_ciaodvd_overall` |
| Industrial validation on KuaiRand-1K | `experiments/KuaiRand_industrial_validation.cpp` | `run_kuairand_industrial_validation` |
| Full two-stage framework fusion | `experiments/Full_two_Stage_fusion.cpp` | `run_full_two_stage_fusion` |
| Stage-1 only experiment | `experiments/Stage1_only.cpp` | `run_stage1_only` |
| Basic 1v1 replacement | `experiments/Stage2_Basic_1v1.cpp` | `run_stage2_basic_1v1` |
| 1v1 replacement with oversampling | `experiments/Stage2_1v1_oversampling.cpp` | `run_stage2_1v1_oversampling` |
| 2v2 fallback replacement with oversampling | `experiments/Stage2_2v2_oversampling.cpp` | `run_stage2_2v2_oversampling` |

For example, on Windows, after building with Visual Studio, the KuaiRand-1K industrial validation experiment can be run as:

```bash
build\\bin\\Release\\run_kuairand_industrial_validation.exe
```

On Linux or macOS, it can be run as:

```bash
./build/bin/run_kuairand_industrial_validation
```

If your local build system places executable files in a different directory, please run the corresponding executable from that directory.

## Notes on Paths

Some experiment files may read processed data from pre-defined relative paths such as:

```text
data/processed/
```

Before running each experiment, please make sure that the required processed files have been generated and placed in the expected directory.

If you use a different data directory, please update the corresponding data path in the relevant experiment entry file or pass the path as a command-line argument if supported.

## Reproducibility Notes

The repository is organized so that each experiment in the paper can be traced to a corresponding C++ entry file. The shared retrieval and evaluation logic is implemented in the header files under `include/`, while the experiment-specific settings and execution logic are provided in `experiments/`.

The preprocessing scripts under `scripts/` are used to convert the raw datasets into the processed files required by the C++ experiments.

## Anonymity

This repository is prepared for double-blind review. It does not contain author names, institutional information, or non-anonymized project identifiers.
