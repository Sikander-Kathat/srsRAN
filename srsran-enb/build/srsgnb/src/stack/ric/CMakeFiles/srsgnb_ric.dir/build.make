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
include srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/depend.make

# Include the progress variables for this target.
include srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/progress.make

# Include the compile flags for this target's objects.
include srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/flags.make

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2_agent.cc.o: srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/flags.make
srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2_agent.cc.o: ../srsgnb/src/stack/ric/e2_agent.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-enb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2_agent.cc.o"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/srsgnb_ric.dir/e2_agent.cc.o -c /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2_agent.cc

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2_agent.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/srsgnb_ric.dir/e2_agent.cc.i"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2_agent.cc > CMakeFiles/srsgnb_ric.dir/e2_agent.cc.i

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2_agent.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/srsgnb_ric.dir/e2_agent.cc.s"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2_agent.cc -o CMakeFiles/srsgnb_ric.dir/e2_agent.cc.s

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2ap_ric_subscription.cc.o: srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/flags.make
srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2ap_ric_subscription.cc.o: ../srsgnb/src/stack/ric/e2ap_ric_subscription.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-enb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2ap_ric_subscription.cc.o"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/srsgnb_ric.dir/e2ap_ric_subscription.cc.o -c /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2ap_ric_subscription.cc

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2ap_ric_subscription.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/srsgnb_ric.dir/e2ap_ric_subscription.cc.i"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2ap_ric_subscription.cc > CMakeFiles/srsgnb_ric.dir/e2ap_ric_subscription.cc.i

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2ap_ric_subscription.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/srsgnb_ric.dir/e2ap_ric_subscription.cc.s"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2ap_ric_subscription.cc -o CMakeFiles/srsgnb_ric.dir/e2ap_ric_subscription.cc.s

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2ap.cc.o: srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/flags.make
srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2ap.cc.o: ../srsgnb/src/stack/ric/e2ap.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-enb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2ap.cc.o"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/srsgnb_ric.dir/e2ap.cc.o -c /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2ap.cc

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2ap.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/srsgnb_ric.dir/e2ap.cc.i"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2ap.cc > CMakeFiles/srsgnb_ric.dir/e2ap.cc.i

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2ap.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/srsgnb_ric.dir/e2ap.cc.s"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2ap.cc -o CMakeFiles/srsgnb_ric.dir/e2ap.cc.s

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm_common.cc.o: srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/flags.make
srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm_common.cc.o: ../srsgnb/src/stack/ric/e2sm_kpm_common.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-enb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm_common.cc.o"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/srsgnb_ric.dir/e2sm_kpm_common.cc.o -c /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2sm_kpm_common.cc

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm_common.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/srsgnb_ric.dir/e2sm_kpm_common.cc.i"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2sm_kpm_common.cc > CMakeFiles/srsgnb_ric.dir/e2sm_kpm_common.cc.i

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm_common.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/srsgnb_ric.dir/e2sm_kpm_common.cc.s"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2sm_kpm_common.cc -o CMakeFiles/srsgnb_ric.dir/e2sm_kpm_common.cc.s

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm.cc.o: srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/flags.make
srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm.cc.o: ../srsgnb/src/stack/ric/e2sm_kpm.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-enb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm.cc.o"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/srsgnb_ric.dir/e2sm_kpm.cc.o -c /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2sm_kpm.cc

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/srsgnb_ric.dir/e2sm_kpm.cc.i"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2sm_kpm.cc > CMakeFiles/srsgnb_ric.dir/e2sm_kpm.cc.i

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/srsgnb_ric.dir/e2sm_kpm.cc.s"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2sm_kpm.cc -o CMakeFiles/srsgnb_ric.dir/e2sm_kpm.cc.s

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm_report_service.cc.o: srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/flags.make
srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm_report_service.cc.o: ../srsgnb/src/stack/ric/e2sm_kpm_report_service.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-enb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm_report_service.cc.o"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/srsgnb_ric.dir/e2sm_kpm_report_service.cc.o -c /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2sm_kpm_report_service.cc

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm_report_service.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/srsgnb_ric.dir/e2sm_kpm_report_service.cc.i"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2sm_kpm_report_service.cc > CMakeFiles/srsgnb_ric.dir/e2sm_kpm_report_service.cc.i

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm_report_service.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/srsgnb_ric.dir/e2sm_kpm_report_service.cc.s"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric/e2sm_kpm_report_service.cc -o CMakeFiles/srsgnb_ric.dir/e2sm_kpm_report_service.cc.s

# Object files for target srsgnb_ric
srsgnb_ric_OBJECTS = \
"CMakeFiles/srsgnb_ric.dir/e2_agent.cc.o" \
"CMakeFiles/srsgnb_ric.dir/e2ap_ric_subscription.cc.o" \
"CMakeFiles/srsgnb_ric.dir/e2ap.cc.o" \
"CMakeFiles/srsgnb_ric.dir/e2sm_kpm_common.cc.o" \
"CMakeFiles/srsgnb_ric.dir/e2sm_kpm.cc.o" \
"CMakeFiles/srsgnb_ric.dir/e2sm_kpm_report_service.cc.o"

# External object files for target srsgnb_ric
srsgnb_ric_EXTERNAL_OBJECTS =

srsgnb/src/stack/ric/libsrsgnb_ric.a: srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2_agent.cc.o
srsgnb/src/stack/ric/libsrsgnb_ric.a: srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2ap_ric_subscription.cc.o
srsgnb/src/stack/ric/libsrsgnb_ric.a: srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2ap.cc.o
srsgnb/src/stack/ric/libsrsgnb_ric.a: srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm_common.cc.o
srsgnb/src/stack/ric/libsrsgnb_ric.a: srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm.cc.o
srsgnb/src/stack/ric/libsrsgnb_ric.a: srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/e2sm_kpm_report_service.cc.o
srsgnb/src/stack/ric/libsrsgnb_ric.a: srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/build.make
srsgnb/src/stack/ric/libsrsgnb_ric.a: srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-enb/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX static library libsrsgnb_ric.a"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && $(CMAKE_COMMAND) -P CMakeFiles/srsgnb_ric.dir/cmake_clean_target.cmake
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/srsgnb_ric.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/build: srsgnb/src/stack/ric/libsrsgnb_ric.a

.PHONY : srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/build

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/clean:
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric && $(CMAKE_COMMAND) -P CMakeFiles/srsgnb_ric.dir/cmake_clean.cmake
.PHONY : srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/clean

srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/depend:
	cd /home/EdgeRIC-A-real-time-RIC/srsran-enb/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/EdgeRIC-A-real-time-RIC/srsran-enb /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsgnb/src/stack/ric /home/EdgeRIC-A-real-time-RIC/srsran-enb/build /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : srsgnb/src/stack/ric/CMakeFiles/srsgnb_ric.dir/depend

