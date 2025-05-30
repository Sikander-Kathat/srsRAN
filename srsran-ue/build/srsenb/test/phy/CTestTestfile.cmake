# CMake generated Testfile for 
# Source directory: /home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy
# Build directory: /home/EdgeRIC-A-real-time-RIC/srsran-ue/build/srsenb/test/phy
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(enb_phy_test_tm1 "enb_phy_test" "--duration=128" "--cell.nof_prb=100" "--tm=1")
set_tests_properties(enb_phy_test_tm1 PROPERTIES  LABELS "lte;srsenb;phy" _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/CMakeLists.txt;631;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;38;add_lte_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;0;")
add_test(enb_phy_test_tm2 "enb_phy_test" "--duration=128" "--cell.nof_prb=100" "--tm=2")
set_tests_properties(enb_phy_test_tm2 PROPERTIES  LABELS "lte;srsenb;phy" _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/CMakeLists.txt;631;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;45;add_lte_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;0;")
add_test(enb_phy_test_tm3 "enb_phy_test" "--duration=128" "--cell.nof_prb=100" "--tm=3")
set_tests_properties(enb_phy_test_tm3 PROPERTIES  LABELS "lte;srsenb;phy" _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/CMakeLists.txt;631;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;52;add_lte_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;0;")
add_test(enb_phy_test_tm4 "enb_phy_test" "--duration=128" "--cell.nof_prb=100" "--tm=4")
set_tests_properties(enb_phy_test_tm4 PROPERTIES  LABELS "lte;srsenb;phy" _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/CMakeLists.txt;631;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;59;add_lte_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;0;")
add_test(enb_phy_test_tm1_ca_pucch3 "enb_phy_test" "--duration=128" "--nof_enb_cells=5" "--ue_cell_list=3,4,0,1,2" "--ack_mode=pucch3" "--cell.nof_prb=6" "--tm=1")
set_tests_properties(enb_phy_test_tm1_ca_pucch3 PROPERTIES  LABELS "lte;srsenb;phy" _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/CMakeLists.txt;631;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;67;add_lte_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;0;")
add_test(enb_phy_test_tm4_ca_pucch3 "enb_phy_test" "--duration=128" "--nof_enb_cells=5" "--ue_cell_list=0,4,3,1,2" "--ack_mode=pucch3" "--cell.nof_prb=6" "--tm=4")
set_tests_properties(enb_phy_test_tm4_ca_pucch3 PROPERTIES  LABELS "lte;srsenb;phy" _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/CMakeLists.txt;631;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;75;add_lte_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;0;")
add_test(enb_phy_test_tm1_ca_cs "enb_phy_test" "--duration=128" "--nof_enb_cells=5" "--ue_cell_list=4,3" "--ack_mode=cs" "--cell.nof_prb=6" "--tm=1")
set_tests_properties(enb_phy_test_tm1_ca_cs PROPERTIES  LABELS "lte;srsenb;phy" _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/CMakeLists.txt;631;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;83;add_lte_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;0;")
add_test(enb_phy_test_tm4_ca_cs "enb_phy_test" "--duration=128" "--nof_enb_cells=5" "--ue_cell_list=1,4" "--ack_mode=cs" "--cell.nof_prb=6" "--tm=4")
set_tests_properties(enb_phy_test_tm4_ca_cs PROPERTIES  LABELS "lte;srsenb;phy" _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/CMakeLists.txt;631;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;91;add_lte_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;0;")
add_test(enb_phy_test_tm1_ca_cs_ho "enb_phy_test" "--duration=1000" "--nof_enb_cells=3" "--ue_cell_list=2,0" "--ack_mode=cs" "--cell.nof_prb=100" "--tm=1" "--rotation=100")
set_tests_properties(enb_phy_test_tm1_ca_cs_ho PROPERTIES  LABELS "lte;srsenb;phy" _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/CMakeLists.txt;631;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;99;add_lte_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;0;")
add_test(enb_phy_test_exceed_nof_carriers "enb_phy_test" "--duration=128" "--nof_enb_cells=6" "--ue_cell_list=1,5" "--ack_mode=cs" "--cell.nof_prb=6" "--tm=4")
set_tests_properties(enb_phy_test_exceed_nof_carriers PROPERTIES  LABELS "lte;srsenb;phy" _BACKTRACE_TRIPLES "/home/EdgeRIC-A-real-time-RIC/srsran-ue/CMakeLists.txt;631;add_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;102;add_lte_test;/home/EdgeRIC-A-real-time-RIC/srsran-ue/srsenb/test/phy/CMakeLists.txt;0;")
