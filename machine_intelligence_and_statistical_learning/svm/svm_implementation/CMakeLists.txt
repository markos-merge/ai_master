# Set the minimum required version of CMake for this project
cmake_minimum_required(VERSION 3.10)

# Define the project name and specify that it's a C++ project
project(hello_rocm CXX)

# Set the C++ standard to C++11 (or higher if needed)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the ROCm SMI header file (rocm_smi.h)
# We hint at the standard installation path on Linux
find_path(ROCM_SMI_INCLUDE_DIR rocm_smi/rocm_smi.h
    HINTS /opt/rocm/include)

# Find the ROCm SMI library (librocm_smi64.so)
find_library(ROCM_SMI_LIBRARY
    NAMES rocm_smi64
    HINTS /opt/rocm/lib)

# If either the include directory or the library is not found, stop with an error
if(NOT ROCM_SMI_INCLUDE_DIR OR NOT ROCM_SMI_LIBRARY)
    message(FATAL_ERROR "ROCm SMI library/headers not found. Ensure ROCm is installed in /opt/rocm.")
endif()

# Add the executable target, built from our source file
add_executable(hello_rocm hello_rocm.cpp)

# Link our executable against the ROCm SMI library
target_link_libraries(hello_rocm PUBLIC ${ROCM_SMI_LIBRARY})
target_include_directories(hello_rocm PUBLIC ${ROCM_SMI_INCLUDE_DIR})

# Add a custom target to build and then run the executable.
# This assumes your executable target is named 'hello_rocm'.
add_custom_target(build_and_run
    # The COMMAND will run the executable after it is built.
    # $<TARGET_FILE:hello_rocm> is a generator expression that gets the path to the executable.
    COMMAND $<TARGET_FILE:hello_rocm>
    # The DEPENDS ensures that the 'hello_rocm' executable is up-to-date before the COMMAND is run.
    DEPENDS hello_rocm
    # USES_TERMINAL ensures the output of your program is shown in the console.
    USES_TERMINAL
    COMMENT "Building and running the hello_rocm executable"
)
