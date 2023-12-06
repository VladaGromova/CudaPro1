# Compiler
CXX = g++
NVCC = nvcc

# Flags
CXXFLAGS = -std=c++11 -Wall
NVCCFLAGS = -arch=sm_75 # Change the architecture according to your GPU

# File names
CPP_FILE = main.cpp
CU_FILES = kernel1.cu kernel2.cu

# Output file name
OUTPUT = executable_name

# Compilation rule for .cpp file
$(OUTPUT): $(CPP_FILE) $(CU_FILES)
    $(NVCC) $(NVCCFLAGS) -o $(OUTPUT) $(CPP_FILE) $(CU_FILES)

.PHONY: clean

clean:
    rm -f $(OUTPUT)
