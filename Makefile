# Compiler
CXX = g++
NVCC = nvcc

# Flags
# CXXFLAGS = -std=c++11 -Wall
# NVCCFLAGS = -arch=sm_75 # Change the architecture according to your GPU

# File names
CPP_FILE = seq.cpp
CU_FILE_1 = parallel.cu
CU_FILE_2 = thrust.cu

# Output file names
OUTPUT_CPP = sequential_cpp
OUTPUT_CU_1 = kernel_cu
OUTPUT_CU_2 = thrust_cu

# Compilation rule for .cpp file
$(OUTPUT_CPP): $(CPP_FILE)
    $(CXX) -o $(OUTPUT_CPP) $(CPP_FILE)

# Compilation rule for first .cu file
$(OUTPUT_CU_1): $(CU_FILE_1)
    $(NVCC) -o $(OUTPUT_CU_1) $(CU_FILE_1)

# Compilation rule for second .cu file
$(OUTPUT_CU_2): $(CU_FILE_2)
    $(NVCC) -o $(OUTPUT_CU_2) $(CU_FILE_2)

.PHONY: clean

clean:
    rm -f $(OUTPUT_CPP) $(OUTPUT_CU_1) $(OUTPUT_CU_2)
