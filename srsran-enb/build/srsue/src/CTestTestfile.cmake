# CMake generated Testfile for 
# Source directory: /home/EdgeRIC-A-real-time-RIC/srsran-enb/srsue/src
# Build directory: /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/srsue/src
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(srsue_version "srsue" "--version")
set_tests_properties(srsue_version PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-enb/srsue/src/CMakeLists.txt;58;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-enb/srsue/src/CMakeLists.txt;0;")
add_test(srsue_help "srsue" "--help")
set_tests_properties(srsue_help PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-enb/srsue/src/CMakeLists.txt;59;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-enb/srsue/src/CMakeLists.txt;0;")
add_test(ue_rf_failure "srsue" "/home/EdgeRIC-A-real-time-RIC/srsran-enb/srsue/ue.conf.example" "--rf.device_name=zmq")
set_tests_properties(ue_rf_failure PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-enb/srsue/src/CMakeLists.txt;64;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-enb/srsue/src/CMakeLists.txt;0;")
add_test(ue_rf_failure_max_channels "srsue" "/home/EdgeRIC-A-real-time-RIC/srsran-enb/srsue/ue.conf.example" "--rf.device_name=zmq" "--rf.nof_antennas=4" "--rat.eutra.nof_carriers=5")
set_tests_properties(ue_rf_failure_max_channels PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-enb/srsue/src/CMakeLists.txt;65;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-enb/srsue/src/CMakeLists.txt;0;")
add_test(ue_rf_failure_exceeds_channels "srsue" "/home/EdgeRIC-A-real-time-RIC/srsran-enb/srsue/ue.conf.example" "--rf.device_name=zmq" "--rf.nof_antennas=5" "--rat.eutra.nof_carriers=5")
set_tests_properties(ue_rf_failure_exceeds_channels PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-enb/srsue/src/CMakeLists.txt;66;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-enb/srsue/src/CMakeLists.txt;0;")
subdirs("phy")
subdirs("stack")
subdirs("test")
