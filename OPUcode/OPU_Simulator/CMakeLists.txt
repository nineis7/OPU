cmake_minimum_required(VERSION 3.1.0)
project (simulator)

set(CMAKE_CXX_FLAGS  "--std=c++11 -g")

find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})
add_library(cnpy SHARED "3rdparty/cnpy/cnpy.cpp")
target_link_libraries(cnpy ${ZLIB_LIBRARIES})

include_directories("./")

set(SOURCES 
  src/driver_bin.cc 
  src/instruction.cc 
  src/accelerator.cc 
  src/vmem.cc 
  src/profiler.cc
  src/bert_activations.cc
  src/cordic.cc
)

add_executable(sim-bin ${SOURCES})

target_link_libraries(sim-bin "-lglog" cnpy)
find_path(GLOG_INCLUDE_DIR glog/logging.h)
target_include_directories(sim-bin PUBLIC ${GLOG_INCLUDE_DIR})


add_executable(txt2bin src/util/txt2bin.cpp)


include_directories(3rdparty/json/single_include)
include_directories(3rdparty/cnpy)

set(SOURCES1 
  src/driver_json.cc 
  src/instruction.cc 
  src/accelerator.cc 
  src/vmem.cc 
  src/profiler.cc
  src/bert_activations.cc
  src/cordic.cc
)
add_executable(sim-json ${SOURCES1})
target_link_libraries(sim-json "-lglog" cnpy)
target_include_directories(sim-json PUBLIC ${GLOG_INCLUDE_DIR})