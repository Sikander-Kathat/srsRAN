/**
 * Copyright 2013-2022 Software Radio Systems Limited
 *
 * This file is part of srsRAN.
 *
 * srsRAN is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * srsRAN is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * A copy of the GNU Affero General Public License can be found in
 * the LICENSE file in the top-level directory of this distribution
 * and at http://www.gnu.org/licenses/.
 *
 */

#ifndef SRSRAN_BUILD_INFO_H_IN
#define SRSRAN_BUILD_INFO_H_IN

#include "srsran/config.h"

# ifdef __cplusplus
extern "C" {
# endif

#ifdef BUILD_TYPE_RELEASE
  static char build_mode[] = "Release";
#else
  #ifdef BUILD_TYPE_DEBUG
    static char build_mode[] = "Debug";
  #else
    #ifdef BUILD_TYPE_RELWITHDEBINFO
      static char build_mode[] = "RelWithDebInfo";
    #else
      static char build_mode[] = "unknown";
    #endif
  #endif
#endif

// the configured build options for srsRAN
static char build_info[] = "commit  on branch ";

SRSRAN_API char* srsran_get_build_info() {
  return build_info;
};

SRSRAN_API char* srsran_get_build_mode() {
  return build_mode;
}

# ifdef __cplusplus
}
# endif

#endif // BUILD_INFO_
