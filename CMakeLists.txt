CMAKE_MINIMUM_REQUIRED(VERSION 3.13.4)

FIND_PACKAGE(deal.II 9.2
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

# Enable a switchable dimension choice
IF (NOT DEFINED DIM)
        SET(DIM 2)
ENDIF()
ADD_DEFINITIONS(-DDIM=${DIM})

# Set the target and the target source
SET( TARGET "elasticity" )
SET( TARGET_SRC
    ${TARGET}.cc
    ${CMAKE_SOURCE_DIR}/source/linear_elasticity/linear_elasticity.cc
    ${CMAKE_SOURCE_DIR}/source/nonlinear_elasticity/nonlinear_elasticity.cc
    ${CMAKE_SOURCE_DIR}/include/adapter/parameters.cc)

DEAL_II_INITIALIZE_CACHED_VARIABLES()

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "RELEASE")
  MESSAGE(STATUS "No build type specified. Building in ${CMAKE_BUILD_TYPE} mode.")
ENDIF()

# Set the include directory and the name of the project
PROJECT(${TARGET})
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

# The multiphysics repo that carries this adapter as deps/dealii-adapter: its
# actuators.hpp holds the servo model (rb::servoStep) the hinge servo runs.
SET(MP_SHARED_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../.." CACHE PATH
    "Directory containing the multiphysics actuators.hpp")
IF(NOT EXISTS "${MP_SHARED_INCLUDE_DIR}/actuators.hpp")
  MESSAGE(FATAL_ERROR "actuators.hpp not found in MP_SHARED_INCLUDE_DIR "
    "(${MP_SHARED_INCLUDE_DIR}); pass -DMP_SHARED_INCLUDE_DIR=<multiphysics repo>.")
ENDIF()
# SYSTEM: deal.II builds with -Wmissing-braces, which the brace-elided
# std::array initialisers in the shared headers would trip.
INCLUDE_DIRECTORIES(SYSTEM ${MP_SHARED_INCLUDE_DIR})

# Set the executable
ADD_EXECUTABLE(${TARGET} ${TARGET_SRC})

# Use deal.II macros for setting up the target
DEAL_II_SETUP_TARGET(${TARGET})
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Query the git information and set it in the source
DEAL_II_QUERY_GIT_INFORMATION()
SET_PROPERTY(TARGET  ${TARGET}  APPEND  PROPERTY  COMPILE_DEFINITIONS
            GIT_BRANCH="${GIT_BRANCH}"
            GIT_REVISION="${GIT_REVISION}"
            GIT_SHORTREV="${GIT_SHORTREV}")

# the environment variable precice_DIR is searched by default
FIND_PACKAGE(precice 3.0
        HINTS ${precice_DIR} ${PRECICE_DIR} $ENV{PRECICE_DIR}
  )
IF(NOT ${precice_FOUND})
   MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of preCICE. ***\n\n"
    "You may want to either pass a flag -Dprecice_DIR=/path/to/precice to cmake\n"
    "(where the path points to the installation prefix or the build directory)\n"
    " or set an environment variable \"precice_DIR\" that contains this path."
    )
ENDIF()
MESSAGE(STATUS "Using the preCICE version found at ${precice_DIR}")
TARGET_LINK_LIBRARIES(${TARGET} precice::precice)

#
# Custom "debug" and "release" make targets:
#
ADD_CUSTOM_TARGET(debug
COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
)

ADD_CUSTOM_TARGET(release
COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
COMMENT "Switch CMAKE_BUILD_TYPE to Release"
)
