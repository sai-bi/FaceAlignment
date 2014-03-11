cmake_minimum_required(VERSION 2.8)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.45.0 COMPONENTS REQUIRED) 
find_package( OpenCV REQUIRED )

set(src_list face.cpp face.h)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
    add_executable(boostTest ${src_list}) 
    target_link_libraries(boostTest ${Boost_LIBRARIES})   
endif()

set(src_list_2 faceCrop.cpp)
add_executable(faceCrop ${src_list_2})
target_link_libraries(faceCrop ${OpenCV_LIBS})

set(src_list_4 face.h face.cpp faceTraining.cpp)
add_executable(faceTraining ${src_list_4})
target_link_libraries(faceTraining ${OpenCV_LIBS})



