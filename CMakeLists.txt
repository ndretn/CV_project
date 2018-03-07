cmake_minimum_required(VERSION 2.8.11)
project(face_recognition)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(OpenCV REQUIRED)
find_package(PCL 1.3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} include)

add_library(face_pose_tree
  src/CRTree.cpp
)

target_link_libraries(face_pose_tree 
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
)

add_library(face_pose_estimator
  src/CRForestEstimator.cpp
)

target_link_libraries(face_pose_estimator 
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	face_pose_tree
)

add_library(face_extraction 
        src/FaceExtractor.cpp
)
target_link_libraries(face_extraction 
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
	face_pose_estimator
)

add_library(face
	src/Face.cpp
)
target_link_libraries(face 
	${OpenCV_LIBS}
)
add_library(pose
	src/Pose.cpp
)
target_link_libraries(pose 
	${OpenCV_LIBS}
	face
)

add_library(covariance_computer
        src/CovarianceComputer.cpp
)
target_link_libraries(covariance_computer
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
        face_extraction
        pose
)

add_library(svm
				src/svm.cpp
)



add_library(svm_wrapper
				src/svm_wrapper.cpp
)
target_link_libraries(svm_wrapper
        svm
)


add_executable(face_recognition
	src/main.cpp
)
target_link_libraries(face_recognition
	${OpenCV_LIBS}
	${PCL_LIBRARIES}
  covariance_computer
	svm_wrapper
)
