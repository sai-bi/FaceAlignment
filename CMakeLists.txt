cmake_minimum_required(VERSION 2.8)

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost 1.45.0 COMPONENTS regex) 

set(src_list boostTest.cpp)

if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS}) 
    add_executable(boostTest ${src_list}) 
    target_link_libraries(boostTest ${Boost_LIBRARIES})   
endif()
