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
CMAKE_SOURCE_DIR = /home/EdgeRIC-A-real-time-RIC/srsran-enb

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/EdgeRIC-A-real-time-RIC/srsran-enb/build

# Include any dependencies generated for this target.
include lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/depend.make

# Include the progress variables for this target.
include lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/progress.make

# Include the compile flags for this target's objects.
include lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/flags.make

lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/gtpu.cc.o: lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/flags.make
lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/gtpu.cc.o: ../lib/src/gtpu/gtpu.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-enb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/gtpu.cc.o"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/lib/src/gtpu && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/srsran_gtpu.dir/gtpu.cc.o -c /home/EdgeRIC-A-real-time-RIC/srsran-enb/lib/src/gtpu/gtpu.cc

lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/gtpu.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/srsran_gtpu.dir/gtpu.cc.i"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/lib/src/gtpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/EdgeRIC-A-real-time-RIC/srsran-enb/lib/src/gtpu/gtpu.cc > CMakeFiles/srsran_gtpu.dir/gtpu.cc.i

lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/gtpu.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/srsran_gtpu.dir/gtpu.cc.s"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/lib/src/gtpu && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/EdgeRIC-A-real-time-RIC/srsran-enb/lib/src/gtpu/gtpu.cc -o CMakeFiles/srsran_gtpu.dir/gtpu.cc.s

# Object files for target srsran_gtpu
srsran_gtpu_OBJECTS = \
"CMakeFiles/srsran_gtpu.dir/gtpu.cc.o"

# External object files for target srsran_gtpu
srsran_gtpu_EXTERNAL_OBJECTS =

lib/src/gtpu/libsrsran_gtpu.a: lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/gtpu.cc.o
lib/src/gtpu/libsrsran_gtpu.a: lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/build.make
lib/src/gtpu/libsrsran_gtpu.a: lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-enb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libsrsran_gtpu.a"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/lib/src/gtpu && $(CMAKE_COMMAND) -P CMakeFiles/srsran_gtpu.dir/cmake_clean_target.cmake
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/lib/src/gtpu && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/srsran_gtpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/build: lib/src/gtpu/libsrsran_gtpu.a

.PHONY : lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/build

lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/clean:
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/lib/src/gtpu && $(CMAKE_COMMAND) -P CMakeFiles/srsran_gtpu.dir/cmake_clean.cmake
.PHONY : lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/clean

lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/depend:
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/EdgeRIC-A-real-time-RIC/srsran-enb /home/EdgeRIC-A-real-time-RIC/srsran-enb/lib/src/gtpu /home/EdgeRIC-A-real-time-RIC/srsran-enb/build /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/lib/src/gtpu /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/src/gtpu/CMakeFiles/srsran_gtpu.dir/depend

