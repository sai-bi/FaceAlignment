cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )

set(src_list ./face.h ./fern.cpp ./fern_cascade.cpp ./function.cpp ./Main.cpp
    ./shape_regressor.cpp)

add_executable(face ${src_list})
target_link_libraries(face ${OpenCV_LIBS})





