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
include srsenb/rtagent/CMakeFiles/edgeric.dir/depend.make

# Include the progress variables for this target.
include srsenb/rtagent/CMakeFiles/edgeric.dir/progress.make

# Include the compile flags for this target's objects.
include srsenb/rtagent/CMakeFiles/edgeric.dir/flags.make

srsenb/rtagent/CMakeFiles/edgeric.dir/edgeric.cpp.o: srsenb/rtagent/CMakeFiles/edgeric.dir/flags.make
srsenb/rtagent/CMakeFiles/edgeric.dir/edgeric.cpp.o: ../srsenb/rtagent/edgeric.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-enb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object srsenb/rtagent/CMakeFiles/edgeric.dir/edgeric.cpp.o"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsenb/rtagent && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/edgeric.dir/edgeric.cpp.o -c /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsenb/rtagent/edgeric.cpp

srsenb/rtagent/CMakeFiles/edgeric.dir/edgeric.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/edgeric.dir/edgeric.cpp.i"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsenb/rtagent && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsenb/rtagent/edgeric.cpp > CMakeFiles/edgeric.dir/edgeric.cpp.i

srsenb/rtagent/CMakeFiles/edgeric.dir/edgeric.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/edgeric.dir/edgeric.cpp.s"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsenb/rtagent && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsenb/rtagent/edgeric.cpp -o CMakeFiles/edgeric.dir/edgeric.cpp.s

srsenb/rtagent/CMakeFiles/edgeric.dir/metrics.pb.cc.o: srsenb/rtagent/CMakeFiles/edgeric.dir/flags.make
srsenb/rtagent/CMakeFiles/edgeric.dir/metrics.pb.cc.o: ../srsenb/rtagent/metrics.pb.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-enb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object srsenb/rtagent/CMakeFiles/edgeric.dir/metrics.pb.cc.o"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsenb/rtagent && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Wno-array-bounds -o CMakeFiles/edgeric.dir/metrics.pb.cc.o -c /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsenb/rtagent/metrics.pb.cc

srsenb/rtagent/CMakeFiles/edgeric.dir/metrics.pb.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/edgeric.dir/metrics.pb.cc.i"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsenb/rtagent && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Wno-array-bounds -E /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsenb/rtagent/metrics.pb.cc > CMakeFiles/edgeric.dir/metrics.pb.cc.i

srsenb/rtagent/CMakeFiles/edgeric.dir/metrics.pb.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/edgeric.dir/metrics.pb.cc.s"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsenb/rtagent && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Wno-array-bounds -S /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsenb/rtagent/metrics.pb.cc -o CMakeFiles/edgeric.dir/metrics.pb.cc.s

srsenb/rtagent/CMakeFiles/edgeric.dir/scheduling_weights.pb.cc.o: srsenb/rtagent/CMakeFiles/edgeric.dir/flags.make
srsenb/rtagent/CMakeFiles/edgeric.dir/scheduling_weights.pb.cc.o: ../srsenb/rtagent/scheduling_weights.pb.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-enb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object srsenb/rtagent/CMakeFiles/edgeric.dir/scheduling_weights.pb.cc.o"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsenb/rtagent && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Wno-array-bounds -o CMakeFiles/edgeric.dir/scheduling_weights.pb.cc.o -c /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsenb/rtagent/scheduling_weights.pb.cc

srsenb/rtagent/CMakeFiles/edgeric.dir/scheduling_weights.pb.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/edgeric.dir/scheduling_weights.pb.cc.i"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsenb/rtagent && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Wno-array-bounds -E /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsenb/rtagent/scheduling_weights.pb.cc > CMakeFiles/edgeric.dir/scheduling_weights.pb.cc.i

srsenb/rtagent/CMakeFiles/edgeric.dir/scheduling_weights.pb.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/edgeric.dir/scheduling_weights.pb.cc.s"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsenb/rtagent && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -Wno-array-bounds -S /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsenb/rtagent/scheduling_weights.pb.cc -o CMakeFiles/edgeric.dir/scheduling_weights.pb.cc.s

# Object files for target edgeric
edgeric_OBJECTS = \
"CMakeFiles/edgeric.dir/edgeric.cpp.o" \
"CMakeFiles/edgeric.dir/metrics.pb.cc.o" \
"CMakeFiles/edgeric.dir/scheduling_weights.pb.cc.o"

# External object files for target edgeric
edgeric_EXTERNAL_OBJECTS =

srsenb/rtagent/libedgeric.a: srsenb/rtagent/CMakeFiles/edgeric.dir/edgeric.cpp.o
srsenb/rtagent/libedgeric.a: srsenb/rtagent/CMakeFiles/edgeric.dir/metrics.pb.cc.o
srsenb/rtagent/libedgeric.a: srsenb/rtagent/CMakeFiles/edgeric.dir/scheduling_weights.pb.cc.o
srsenb/rtagent/libedgeric.a: srsenb/rtagent/CMakeFiles/edgeric.dir/build.make
srsenb/rtagent/libedgeric.a: srsenb/rtagent/CMakeFiles/edgeric.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-enb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX static library libedgeric.a"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsenb/rtagent && $(CMAKE_COMMAND) -P CMakeFiles/edgeric.dir/cmake_clean_target.cmake
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsenb/rtagent && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/edgeric.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
srsenb/rtagent/CMakeFiles/edgeric.dir/build: srsenb/rtagent/libedgeric.a

.PHONY : srsenb/rtagent/CMakeFiles/edgeric.dir/build

srsenb/rtagent/CMakeFiles/edgeric.dir/clean:
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsenb/rtagent && $(CMAKE_COMMAND) -P CMakeFiles/edgeric.dir/cmake_clean.cmake
.PHONY : srsenb/rtagent/CMakeFiles/edgeric.dir/clean

srsenb/rtagent/CMakeFiles/edgeric.dir/depend:
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/EdgeRIC-A-real-time-RIC/srsran-enb /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsenb/rtagent /home/EdgeRIC-A-real-time-RIC/srsran-enb/build /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsenb/rtagent /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsenb/rtagent/CMakeFiles/edgeric.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : srsenb/rtagent/CMakeFiles/edgeric.dir/depend

