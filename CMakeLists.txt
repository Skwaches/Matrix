cmake_minimum_required(VERSION 4.4)
project(Matrix LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(library Matrix)

# Dynamically use CUDA if available.
option(CUDA_ON "Attempt to use the GPU for matrix multiplication." ON)
set(CUDA_AVAILABLE FALSE PARENT_SCOPE)
if(${CUDA_ON})
	include(CheckLanguage)
	check_language(CUDA)
	if(CMAKE_CUDA_COMPILER)
		enable_language(CUDA)
		set(CUDA_AVAILABLE TRUE)
	else()
		message(WARNING "CUDA compiler has not been found. Attempting CPU only build.")
	endif()
endif()

# Add Library
add_library( "${library}"
	src/matrix.cpp
	src/detect.cpp
	$<$<BOOL:${CUDA_AVAILABLE}>:src/kernel.cu>
	$<$<BOOL:${CUDA_AVAILABLE}>:src/matrix.cu>
)
add_library("${library}::${library}" ALIAS "${library}")

# Include directories
target_include_directories("${library}" PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_definitions("${library}" PRIVATE 
	$<$<BOOL:${CUDA_AVAILABLE}>:CUDA_AVAILABLE>)

# Allow cuda compilation.
if (${CUDA_AVAILABLE})
	find_package(CUDAToolkit)
	set_property(TARGET "${library}"
		PROPERTY
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_RESOLVE_DEVICE_SYMBOLS ON
	)
	target_link_libraries("${library}" PRIVATE CUDA::cudart_static)
endif()

if("${PROJECT_IS_TOP_LEVEL}")
	message(STATUS "Writing compile_commands.json for you ")
	set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()
