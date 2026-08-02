cmake_minimum_required(VERSION 4.4)
project(Matrix LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(library Matrix)

# Add Library
add_library( "${library}" 	
	src/matrix.cpp
	src/matrix.cu
	src/kernel.cu
)
add_library("${library}::${library}" ALIAS "${library}")

# Include directories
target_include_directories("${library}" PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
# Allow cuda compilation.
set_property(TARGET "${library}"
	PROPERTY CMAKE_CUDA_SEPARABLE_COMPILATION ON)

if("${PROJECT_IS_TOP_LEVEL}")
	set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()
