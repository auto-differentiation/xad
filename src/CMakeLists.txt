##############################################################################
#   
#  Cmake file for XAD library and headers
#
#  This file is part of XAD, a comprehensive C++ library for
#  automatic differentiation.
#
#  Copyright (C) 2010-2024 Xcelerit Computing Ltd.
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#   
##############################################################################

configure_file(XAD/Version.hpp.in 
               ${CMAKE_CURRENT_BINARY_DIR}/XAD/Version.hpp
               @ONLY)
configure_file(XAD/Config.hpp.in 
               ${CMAKE_CURRENT_BINARY_DIR}/XAD/Config.hpp
               @ONLY)

set(public_headers
    XAD/BinaryDerivativeImpl.hpp
    XAD/BinaryExpr.hpp
    XAD/BinaryFunctors.hpp
    XAD/BinaryMathFunctors.hpp
    XAD/BinaryOperatorMacros.hpp
    XAD/BinaryOperators.hpp
    XAD/CheckpointCallback.hpp
    XAD/ChunkContainer.hpp
    XAD/Complex.hpp
    XAD/Exceptions.hpp
    XAD/Expression.hpp
    XAD/Hessian.hpp	
    XAD/Interface.hpp	
    XAD/Jacobian.hpp
    XAD/Literals.hpp
    XAD/Macros.hpp
    XAD/MathFunctions.hpp
    XAD/OperationsContainer.hpp
    XAD/OperationsContainerPaired.hpp
    XAD/ReusableRange.hpp
    XAD/StdCompatibility.hpp
    XAD/Tape.hpp
    XAD/TapeContainer.hpp
    XAD/Traits.hpp
    XAD/TypeTraits.hpp
    XAD/UnaryExpr.hpp
    XAD/UnaryFunctors.hpp
    XAD/UnaryMathFunctors.hpp
    XAD/UnaryOperatorMacros.hpp
    XAD/UnaryOperators.hpp
    XAD/XAD.hpp
    # generated files
    ${CMAKE_CURRENT_BINARY_DIR}/XAD/Version.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/XAD/Config.hpp
)

                
set(srcfiles Tape.cpp ${public_headers})
if (MSVC)
    list(APPEND srcfiles ${PROJECT_SOURCE_DIR}/XAD.natvis)
endif()

xad_add_library(xad STATIC ${srcfiles})
target_include_directories(xad PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
    "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>"
)
set_target_properties(xad PROPERTIES VERSION "${XAD_VERSION}")

# Install targets
install(FILES ${public_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/XAD)
install(TARGETS xad 
    EXPORT XADTargets
    RUNTIME DESTINATION 
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")   # clang-cl does not generate PDB files, but rather embeds them
    # install PDB files
    get_target_property(xad_pdb_name xad COMPILE_PDB_NAME)
    get_target_property(xad_pdb_name_debug xad COMPILE_PDB_NAME_DEBUG)
    get_target_property(xad_pdb_output_dir xad PDB_OUTPUT_DIRECTORY)
    get_property(multiconf GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(multiconf)
        install(FILES "${xad_pdb_output_dir}/\${CMAKE_INSTALL_CONFIG_NAME}/$<$<CONFIG:Debug>:${xad_pdb_name_debug}>$<$<NOT:$<CONFIG:Debug>>:${xad_pdb_name}>.pdb"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            CONFIGURATIONS Debug RelWithDebInfo
        )
    else()
        install(FILES "${xad_pdb_output_dir}/$<$<CONFIG:Debug>:${xad_pdb_name_debug}>$<$<NOT:$<CONFIG:Debug>>:${xad_pdb_name}>.pdb"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            CONFIGURATIONS Debug RelWithDebInfo
        )
    endif()
endif()
if(MSVC AND MSVC_VERSION EQUAL 1900)
    # disable forceinline warning in tape.cpp compilation in VS 2015
    target_compile_options(xad PRIVATE /wd4714)
    # disable warning about too long expression symbols in VS 2015
    target_compile_options(xad PUBLIC /wd4503)
endif()

# install CMake export configuration files
include(CMakePackageConfigHelpers)

set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(version_file "${generated_dir}/XADConfigVersion.cmake")
set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/XAD")

write_basic_package_version_file(${version_file}
    VERSION ${XAD_VERSION} 
    COMPATIBILITY SameMajorVersion
)
export(EXPORT XADTargets
    FILE "${PROJECT_BINARY_DIR}/cmake/XADTargets.cmake"
    NAMESPACE XAD::
)
install(EXPORT XADTargets
    FILE XADTargets.cmake
    NAMESPACE XAD::
    DESTINATION ${config_install_dir}
)
configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/XADConfig.cmake.in"
    "${generated_dir}/XADConfig.cmake" 
    INSTALL_DESTINATION ${config_install_dir}
)
install(FILES 
            "${generated_dir}/XADConfigVersion.cmake"
            "${generated_dir}/XADConfig.cmake" 
        DESTINATION 
            ${config_install_dir}
)
