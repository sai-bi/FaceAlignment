cmake_minimum_required(VERSION 2.8)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.45.0 COMPONENTS regex) 
find_package( OpenCV REQUIRED )

set(src_list face.cpp face.h)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
    add_executable(boostTest ${src_list}) 
    target_link_libraries(boostTest ${Boost_LIBRARIES})   
endif()

set(src_list_1 faceDetector.cpp)
add_executable(facedetector ${src_list_1})
target_link_libraries(facedetector ${OpenCV_LIBS})


