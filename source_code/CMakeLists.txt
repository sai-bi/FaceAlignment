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


set(src_list_2 ./face.h ./fern.cpp ./fern_cascade.cpp ./function.cpp ./face_crop.cpp
    ./shape_regressor.cpp)
add_executable(face_crop.out ${src_list_2})
target_link_libraries(face_crop.out ${OpenCV_LIBS})


set(src_list_3 ./face.h ./fern.cpp ./fern_cascade.cpp ./function.cpp ./cofw_train.cpp
    ./shape_regressor.cpp)
add_executable(cofw.out ${src_list_3})
target_link_libraries(cofw.out ${OpenCV_LIBS})

set(src_list_4 ./face.h ./fern.cpp ./fern_cascade.cpp ./function.cpp ./cofw_test.cpp
    ./shape_regressor.cpp)
add_executable(cofw_test.out ${src_list_4})
target_link_libraries(cofw_test.out ${OpenCV_LIBS})

