
execute_process(
COMMAND git rev-parse --abbrev-ref HEAD
WORKING_DIRECTORY "/home/EdgeRIC-A-real-time-RIC/srsran-enb"
OUTPUT_VARIABLE GIT_BRANCH
OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
COMMAND git log -1 --format=%h
WORKING_DIRECTORY "/home/EdgeRIC-A-real-time-RIC/srsran-enb"
OUTPUT_VARIABLE GIT_COMMIT_HASH
OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "Generating build_info.h")
configure_file(
  /home/EdgeRIC-A-real-time-RIC/srsran-enb/lib/include/srsran/build_info.h.in
  /home/EdgeRIC-A-real-time-RIC/srsran-enb/build/lib/include/srsran/build_info.h
)
