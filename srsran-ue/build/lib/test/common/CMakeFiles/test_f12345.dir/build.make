# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/EdgeRIC-A-real-time-RIC/srsran-ue

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/EdgeRIC-A-real-time-RIC/srsran-ue/build

# Include any dependencies generated for this target.
include lib/test/common/CMakeFiles/test_f12345.dir/depend.make

# Include the progress variables for this target.
include lib/test/common/CMakeFiles/test_f12345.dir/progress.make

# Include the compile flags for this target's objects.
include lib/test/common/CMakeFiles/test_f12345.dir/flags.make

lib/test/common/CMakeFiles/test_f12345.dir/test_f12345.cc.o: lib/test/common/CMakeFiles/test_f12345.dir/flags.make
lib/test/common/CMakeFiles/test_f12345.dir/test_f12345.cc.o: ../lib/test/common/test_f12345.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-ue/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/test/common/CMakeFiles/test_f12345.dir/test_f12345.cc.o"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-ue/build/lib/test/common && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_f12345.dir/test_f12345.cc.o -c /home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/test/common/test_f12345.cc

lib/test/common/CMakeFiles/test_f12345.dir/test_f12345.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_f12345.dir/test_f12345.cc.i"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-ue/build/lib/test/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/test/common/test_f12345.cc > CMakeFiles/test_f12345.dir/test_f12345.cc.i

lib/test/common/CMakeFiles/test_f12345.dir/test_f12345.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_f12345.dir/test_f12345.cc.s"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-ue/build/lib/test/common && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/test/common/test_f12345.cc -o CMakeFiles/test_f12345.dir/test_f12345.cc.s

# Object files for target test_f12345
test_f12345_OBJECTS = \
"CMakeFiles/test_f12345.dir/test_f12345.cc.o"

# External object files for target test_f12345
test_f12345_EXTERNAL_OBJECTS =

lib/test/common/test_f12345: lib/test/common/CMakeFiles/test_f12345.dir/test_f12345.cc.o
lib/test/common/test_f12345: lib/test/common/CMakeFiles/test_f12345.dir/build.make
lib/test/common/test_f12345: lib/src/common/libsrsran_common.a
lib/test/common/test_f12345: lib/src/phy/libsrsran_phy.a
lib/test/common/test_f12345: /usr/lib/x86_64-linux-gnu/libfftw3f.so
lib/test/common/test_f12345: lib/src/support/libsupport.a
lib/test/common/test_f12345: lib/src/srslog/libsrslog.a
lib/test/common/test_f12345: /usr/lib/x86_64-linux-gnu/libmbedcrypto.so
lib/test/common/test_f12345: lib/test/common/CMakeFiles/test_f12345.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-ue/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_f12345"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-ue/build/lib/test/common && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_f12345.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/test/common/CMakeFiles/test_f12345.dir/build: lib/test/common/test_f12345

.PHONY : lib/test/common/CMakeFiles/test_f12345.dir/build

lib/test/common/CMakeFiles/test_f12345.dir/clean:
	cd /home/EdgeRIC-A-real-time-RIC/srsran-ue/build/lib/test/common && $(CMAKE_COMMAND) -P CMakeFiles/test_f12345.dir/cmake_clean.cmake
.PHONY : lib/test/common/CMakeFiles/test_f12345.dir/clean

lib/test/common/CMakeFiles/test_f12345.dir/depend:
	cd /home/EdgeRIC-A-real-time-RIC/srsran-ue/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/EdgeRIC-A-real-time-RIC/srsran-ue /home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/test/common /home/EdgeRIC-A-real-time-RIC/srsran-ue/build /home/EdgeRIC-A-real-time-RIC/srsran-ue/build/lib/test/common /home/EdgeRIC-A-real-time-RIC/srsran-ue/build/lib/test/common/CMakeFiles/test_f12345.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/test/common/CMakeFiles/test_f12345.dir/depend

