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
include lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/depend.make

# Include the progress variables for this target.
include lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/progress.make

# Include the compile flags for this target's objects.
include lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/flags.make

lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/pdcp_lte_test_status_report.cc.o: lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/flags.make
lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/pdcp_lte_test_status_report.cc.o: ../lib/test/pdcp/pdcp_lte_test_status_report.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-ue/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/pdcp_lte_test_status_report.cc.o"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-ue/build/lib/test/pdcp && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/pdcp_lte_test_status_report.dir/pdcp_lte_test_status_report.cc.o -c /home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/test/pdcp/pdcp_lte_test_status_report.cc

lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/pdcp_lte_test_status_report.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pdcp_lte_test_status_report.dir/pdcp_lte_test_status_report.cc.i"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-ue/build/lib/test/pdcp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/test/pdcp/pdcp_lte_test_status_report.cc > CMakeFiles/pdcp_lte_test_status_report.dir/pdcp_lte_test_status_report.cc.i

lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/pdcp_lte_test_status_report.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pdcp_lte_test_status_report.dir/pdcp_lte_test_status_report.cc.s"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-ue/build/lib/test/pdcp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/test/pdcp/pdcp_lte_test_status_report.cc -o CMakeFiles/pdcp_lte_test_status_report.dir/pdcp_lte_test_status_report.cc.s

# Object files for target pdcp_lte_test_status_report
pdcp_lte_test_status_report_OBJECTS = \
"CMakeFiles/pdcp_lte_test_status_report.dir/pdcp_lte_test_status_report.cc.o"

# External object files for target pdcp_lte_test_status_report
pdcp_lte_test_status_report_EXTERNAL_OBJECTS =

lib/test/pdcp/pdcp_lte_test_status_report: lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/pdcp_lte_test_status_report.cc.o
lib/test/pdcp/pdcp_lte_test_status_report: lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/build.make
lib/test/pdcp/pdcp_lte_test_status_report: lib/src/pdcp/libsrsran_pdcp.a
lib/test/pdcp/pdcp_lte_test_status_report: lib/src/common/libsrsran_common.a
lib/test/pdcp/pdcp_lte_test_status_report: lib/src/phy/libsrsran_phy.a
lib/test/pdcp/pdcp_lte_test_status_report: /usr/lib/x86_64-linux-gnu/libfftw3f.so
lib/test/pdcp/pdcp_lte_test_status_report: lib/src/support/libsupport.a
lib/test/pdcp/pdcp_lte_test_status_report: lib/src/srslog/libsrslog.a
lib/test/pdcp/pdcp_lte_test_status_report: /usr/lib/x86_64-linux-gnu/libmbedcrypto.so
lib/test/pdcp/pdcp_lte_test_status_report: lib/src/asn1/libsrsran_asn1.a
lib/test/pdcp/pdcp_lte_test_status_report: lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/EdgeRIC-A-real-time-RIC/srsran-ue/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pdcp_lte_test_status_report"
	cd /home/EdgeRIC-A-real-time-RIC/srsran-ue/build/lib/test/pdcp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pdcp_lte_test_status_report.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/build: lib/test/pdcp/pdcp_lte_test_status_report

.PHONY : lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/build

lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/clean:
	cd /home/EdgeRIC-A-real-time-RIC/srsran-ue/build/lib/test/pdcp && $(CMAKE_COMMAND) -P CMakeFiles/pdcp_lte_test_status_report.dir/cmake_clean.cmake
.PHONY : lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/clean

lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/depend:
	cd /home/EdgeRIC-A-real-time-RIC/srsran-ue/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/EdgeRIC-A-real-time-RIC/srsran-ue /home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/test/pdcp /home/EdgeRIC-A-real-time-RIC/srsran-ue/build /home/EdgeRIC-A-real-time-RIC/srsran-ue/build/lib/test/pdcp /home/EdgeRIC-A-real-time-RIC/srsran-ue/build/lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/test/pdcp/CMakeFiles/pdcp_lte_test_status_report.dir/depend

