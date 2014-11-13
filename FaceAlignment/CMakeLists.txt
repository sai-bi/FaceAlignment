cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(source ./FaceAlignment.h ./FernCascade.cpp ./Fern.cpp ./ShapeRegressor.cpp ./Utils.cpp ./TrainDemo.cpp)
add_executable(TrainDemo.out ${source})
target_link_libraries(TrainDemo.out ${OpenCV_LIBS})

set(source ./FaceAlignment.h ./FernCascade.cpp ./Fern.cpp ./ShapeRegressor.cpp ./Utils.cpp ./TestDemo.cpp)
add_executable(TestDemo.out ${source})
target_link_libraries(TestDemo.out ${OpenCV_LIBS})


