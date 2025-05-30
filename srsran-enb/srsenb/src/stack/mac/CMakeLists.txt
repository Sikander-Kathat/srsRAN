#
# Copyright 2013-2023 Software Radio Systems Limited
#
# This file is part of srsRAN
#
# srsRAN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# srsRAN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# A copy of the GNU Affero General Public License can be found in
# the LICENSE file in the top-level directory of this distribution
# and at http://www.gnu.org/licenses/.
#

add_subdirectory(common)

add_subdirectory(schedulers)

# Find ZeroMQ if the option is enabled
if(ENABLE_ZEROMQ)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(ZeroMQ REQUIRED libzmq)

  # Include directories for ZeroMQ
  include_directories(${ZeroMQ_INCLUDE_DIRS})
endif()

set(SOURCES mac.cc ue.cc sched.cc sched_carrier.cc sched_grid.cc sched_ue_ctrl/sched_harq.cc sched_ue.cc
            sched_ue_ctrl/sched_lch.cc sched_ue_ctrl/sched_ue_cell.cc sched_ue_ctrl/sched_dl_cqi.cc
            sched_phy_ch/sf_cch_allocator.cc sched_phy_ch/sched_dci.cc sched_phy_ch/sched_phy_resource.cc
            sched_helpers.cc)
add_library(srsenb_mac STATIC ${SOURCES} $<TARGET_OBJECTS:mac_schedulers>)
target_link_libraries(srsenb_mac srsenb_mac_common edgeric)

# Link ZeroMQ to the srsenb_mac library if enabled
if(ENABLE_ZEROMQ)
  target_link_libraries(srsenb_mac ${ZeroMQ_LIBRARIES})
endif()