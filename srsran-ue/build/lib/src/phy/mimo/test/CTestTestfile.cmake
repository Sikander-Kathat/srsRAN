# CMake generated Testfile for 
# Source directory: /home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test
# Build directory: /home/EdgeRIC-A-real-time-RIC/srsran-ue/build/lib/src/phy/mimo/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(layermap_single "layermap_test" "-n" "1000" "-m" "p0" "-c" "1" "-l" "1")
set_tests_properties(layermap_single PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;28;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(layermap_diversity_2 "layermap_test" "-n" "1000" "-m" "div" "-c" "1" "-l" "2")
set_tests_properties(layermap_diversity_2 PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;30;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(layermap_diversity_4 "layermap_test" "-n" "1000" "-m" "div" "-c" "1" "-l" "4")
set_tests_properties(layermap_diversity_4 PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;31;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(layermap_multiplex_11 "layermap_test" "-n" "1000" "-m" "mux" "-c" "1" "-l" "1")
set_tests_properties(layermap_multiplex_11 PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;33;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(layermap_multiplex_12 "layermap_test" "-n" "1000" "-m" "mux" "-c" "1" "-l" "2")
set_tests_properties(layermap_multiplex_12 PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;34;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(layermap_multiplex_13 "layermap_test" "-n" "1002" "-m" "mux" "-c" "1" "-l" "3")
set_tests_properties(layermap_multiplex_13 PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;35;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(layermap_multiplex_14 "layermap_test" "-n" "1000" "-m" "mux" "-c" "1" "-l" "4")
set_tests_properties(layermap_multiplex_14 PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;36;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(layermap_multiplex_22 "layermap_test" "-n" "1000" "-m" "mux" "-c" "2" "-l" "2")
set_tests_properties(layermap_multiplex_22 PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;38;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(layermap_multiplex_23 "layermap_test" "-n" "1002" "-m" "mux" "-c" "2" "-l" "3")
set_tests_properties(layermap_multiplex_23 PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;39;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(layermap_multiplex_24 "layermap_test" "-n" "1000" "-m" "mux" "-c" "2" "-l" "4")
set_tests_properties(layermap_multiplex_24 PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;40;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(precoding_single "precoding_test" "-n" "1000" "-m" "p0")
set_tests_properties(precoding_single PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;50;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(precoding_diversity2 "precoding_test" "-n" "1000" "-m" "div" "-l" "2" "-p" "2")
set_tests_properties(precoding_diversity2 PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;51;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(precoding_diversity4 "precoding_test" "-n" "1024" "-m" "div" "-l" "4" "-p" "4")
set_tests_properties(precoding_diversity4 PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;52;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(precoding_cdd_2x2_zf "precoding_test" "-m" "cdd" "-l" "2" "-p" "2" "-r" "2" "-n" "14000" "-d" "zf")
set_tests_properties(precoding_cdd_2x2_zf PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;54;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(precoding_cdd_2x2_mmse "precoding_test" "-m" "cdd" "-l" "2" "-p" "2" "-r" "2" "-n" "14000" "-d" "mmse")
set_tests_properties(precoding_cdd_2x2_mmse PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;55;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(precoding_multiplex_1l_cb0 "precoding_test" "-m" "mux" "-l" "1" "-p" "2" "-r" "2" "-n" "14000" "-c" "0")
set_tests_properties(precoding_multiplex_1l_cb0 PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;57;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(precoding_multiplex_1l_cb1 "precoding_test" "-m" "mux" "-l" "1" "-p" "2" "-r" "2" "-n" "14000" "-c" "1")
set_tests_properties(precoding_multiplex_1l_cb1 PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;58;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(precoding_multiplex_1l_cb2 "precoding_test" "-m" "mux" "-l" "1" "-p" "2" "-r" "2" "-n" "14000" "-c" "2")
set_tests_properties(precoding_multiplex_1l_cb2 PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;59;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(precoding_multiplex_1l_cb3 "precoding_test" "-m" "mux" "-l" "1" "-p" "2" "-r" "2" "-n" "14000" "-c" "3")
set_tests_properties(precoding_multiplex_1l_cb3 PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;60;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(precoding_multiplex_2l_cb0_zf "precoding_test" "-m" "mux" "-l" "2" "-p" "2" "-r" "2" "-n" "14000" "-c" "0" "-d" "zf")
set_tests_properties(precoding_multiplex_2l_cb0_zf PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;62;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(precoding_multiplex_2l_cb1_zf "precoding_test" "-m" "mux" "-l" "2" "-p" "2" "-r" "2" "-n" "14000" "-c" "1" "-d" "zf")
set_tests_properties(precoding_multiplex_2l_cb1_zf PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;63;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(precoding_multiplex_2l_cb2_zf "precoding_test" "-m" "mux" "-l" "2" "-p" "2" "-r" "2" "-n" "14000" "-c" "2" "-d" "zf")
set_tests_properties(precoding_multiplex_2l_cb2_zf PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;64;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(precoding_multiplex_2l_cb0_mmse "precoding_test" "-m" "mux" "-l" "2" "-p" "2" "-r" "2" "-n" "14000" "-c" "0" "-d" "mmse")
set_tests_properties(precoding_multiplex_2l_cb0_mmse PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;67;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(precoding_multiplex_2l_cb1_mmse "precoding_test" "-m" "mux" "-l" "2" "-p" "2" "-r" "2" "-n" "14000" "-c" "1" "-d" "mmse")
set_tests_properties(precoding_multiplex_2l_cb1_mmse PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;68;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(precoding_multiplex_2l_cb2_mmse "precoding_test" "-m" "mux" "-l" "2" "-p" "2" "-r" "2" "-n" "14000" "-c" "2" "-d" "mmse")
set_tests_properties(precoding_multiplex_2l_cb2_mmse PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;69;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
add_test(pmi_select_test "pmi_select_test")
set_tests_properties(pmi_select_test PROPERTIES  _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;78;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/lib/src/phy/mimo/test/CMakeLists.txt;0;")
