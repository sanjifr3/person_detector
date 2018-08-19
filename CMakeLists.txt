cmake_minimum_required(VERSION 2.8.3)
project(person_detector)

#add_compile_options(-std=c++14)

# Set c++11 cmake flags
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

# Find CUDA
FIND_PACKAGE(CUDA)
if (CUDA_FOUND)
  find_package(CUDA REQUIRED)
  message(STATUS "CUDA Version: ${CUDA_VERSION_STRINGS}")
  message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
  set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    -gencode arch=compute_30,code=sm_30
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=[sm_50,compute_50]
    -gencode arch=compute_52,code=[sm_52,compute_52]
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_62,code=sm_62
  )
  add_definitions(-DGPU)
else()
  list(APPEND LIBRARIES "m")
endif()

# Find rquired packeges
find_package(Boost REQUIRED COMPONENTS thread system)
find_package(PCL 1.7 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(dlib REQUIRED)
# Find OpenCV
find_package (OpenCV 3.4.0 EXACT REQUIRED 
	NO_MODULE  # Should be optional, tells CMake to use config mode
	PATHS /usr/local # Look here
	NO_DEFAULT_PATH # And don't look anywhere else
)

include_directories(${OpenCV_INCLUDE_DIRS})
find_package(catkin REQUIRED
  COMPONENTS
    cv_bridge
    roscpp
    rospy
    std_msgs
    sensor_msgs
    actionlib
    image_transport
    dynamic_reconfigure
  	pcl_conversions
	  pcl_ros
	  image_utils
	  object_detector
)

# Enable OPENCV in darknet
add_definitions(-DOPENCV)
add_definitions(-O4 -g)

add_message_files(
  DIRECTORY msg
  FILES faceDetails.msg
        dlibRect.msg
)

add_service_files(
  DIRECTORY srv
  FILES face_recognition.srv
)

generate_messages(
  DEPENDENCIES
    std_msgs
    sensor_msgs
	  actionlib_msgs
)

generate_dynamic_reconfigure_options(
  cfg/FaceDetectorRQT.cfg
)

catkin_package(
  INCLUDE_DIRS
    include
  LIBRARIES fd_lib
  CATKIN_DEPENDS
    cv_bridge
    roscpp
    pcl_conversions
    dynamic_reconfigure
    pcl_ros
    std_msgs
    image_transport
  DEPENDS
    Boost
)

include_directories(
	include
	SYSTEM
	${catkin_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	${image_geometry_INCLUDE_DIRS}
	${EIGEN_INCLUDE_DIRS}
)

link_directories(
	${PCL_LIBRARY_DIRS}
	${catkin_LIBRARY_DIRS}
	${image_geometry_LIBRARY_DIRS}
	${OpenCV_LIBRARY_DIRS}
	${Boost_LIBRARY_DIRS}
)

add_definitions(
	${PCL_DEFINITIONS}
)

if (CUDA_FOUND)
  link_directories(/usr/local/cuda/lib64)
  
  cuda_add_library(fd_lib src/libraries/FaceDetector.cpp)
else()                   
  add_library(fd_lib src/libraries/FaceDetector.cpp)
endif()

target_link_libraries(fd_lib utils_lib im_proc_lib dlib)
add_dependencies(fd_lib ${PROJECT_NAME}_generate_messages_cpp ${PROJECT_NAME}_gencfg)
