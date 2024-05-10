project(average_3d_ffd)
cmake_minimum_required(VERSION 2.8)

# The directory for compiled executable files
set(EXECUTABLE_OUTPUT_PATH /work/users/y/u/yuukias/Heart_pipeline/third_party/bin)

# MIRTK
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g")
set(MIRTK_SOURCE_DIR /nas/longleaf/apps/mirtk/2.0.0/MIRTK)
set(MIRTK_BINARY_DIR /nas/longleaf/apps/mirtk/2.0.0)

include_directories(${MIRTK_SOURCE_DIR}/Modules/Common/include)
include_directories(${MIRTK_SOURCE_DIR}/Modules/Image/include)
include_directories(${MIRTK_SOURCE_DIR}/Modules/ImageIO/include)
include_directories(${MIRTK_SOURCE_DIR}/Modules/Numerics/include)
include_directories(${MIRTK_SOURCE_DIR}/Modules/PointSet/include)
include_directories(${MIRTK_SOURCE_DIR}/Modules/Registration/include)
include_directories(${MIRTK_SOURCE_DIR}/Modules/Transformation/include)
include_directories(${MIRTK_BINARY_DIR}/include)

link_directories(${MIRTK_BINARY_DIR}/lib)
link_libraries(MIRTKCommon MIRTKNumerics MIRTKImage MIRTKIO MIRTKPointSet MIRTKRegistration MIRTKTransformation)

# VTK
find_package(VTK REQUIRED)
# include(${VTK_USE_FILE})
include_directories(${VTK_INCLUDE_DIRS})
link_libraries(${VTK_LIBRARIES})

# Executables
ADD_EXECUTABLE(average_3d_ffd average_3d_ffd.cc)
