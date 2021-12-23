cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dr_cpp)

ENABLE_TESTING()

find_package(Torch REQUIRED)
find_package(Boost COMPONENTS system thread)


add_library(dr-mvsnet STATIC src/dr_mvsnet.cpp)
target_include_directories(dr-mvsnet PUBLIC src/dr_mvsnet)
target_link_libraries(dr-mvsnet LINK_PRIVATE "${TORCH_LIBRARIES}" boost_system boost_thread)
set_property(TARGET dr-mvsnet PROPERTY CXX_STANDARD 14)

add_executable(dr_mvsnet_test src/dr_mvsnet_test.cpp)
target_link_libraries(dr_mvsnet_test dr-mvsnet)

add_test(NAME test_model_on_data
        COMMAND ./dr_mvsnet_test ../data/model.pt ../data/sample_inputs.pt)
