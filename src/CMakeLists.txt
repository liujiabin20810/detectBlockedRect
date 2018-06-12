cmake_minimum_required(VERSION 2.8)
 
PROJECT(detectBDRect)
 
find_package(OpenCV REQUIRED)

include_directories(detectBDRect ${OpenCV_INCLUDE_DIRS})

file(GLOB src
	main.cpp
	detectBlockedRect.cpp
)

file(GLOB hdr
	detectBlockedRect.hpp
)

add_executable(detectBDRect ${src} ${hdr})
target_link_libraries(detectBDRect ${OpenCV_LIBS})