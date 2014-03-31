cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )

set(src_list ./face.h ./fern.cpp ./fern_cascade.cpp ./function.cpp ./Main.cpp
    ./shape_regressor.cpp)

add_executable(face.out ${src_list})
target_link_libraries(face.out ${OpenCV_LIBS})


set(src_list_1 ./face.h ./fern.cpp ./fern_cascade.cpp ./function.cpp ./test.cpp
    ./shape_regressor.cpp)
add_executable(test.out ${src_list_1})
target_link_libraries(test.out ${OpenCV_LIBS})


set(src_list_2 ./face_crop.cpp)
add_executable(face_crop.out ${src_list_2})
target_link_libraries(face_crop.out ${OpenCV_LIBS})

