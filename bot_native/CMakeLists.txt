cmake_minimum_required(VERSION 3.20)
project(kalshi_bot_native LANGUAGES CXX)

if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)  # Use Boost's own CMake config (Boost ≥ 1.70 ships it).
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)

# ── deps ───────────────────────────────────────────────────────────────
# Homebrew's openssl@3 isn't on the default search path on Apple Silicon.
if(APPLE)
  set(OPENSSL_ROOT_DIR "/opt/homebrew/opt/openssl@3" CACHE PATH "openssl prefix")
endif()
find_package(OpenSSL REQUIRED)
find_package(Boost 1.80 REQUIRED CONFIG)
find_package(nlohmann_json REQUIRED)
find_package(Threads REQUIRED)

# ── library: core bot logic ────────────────────────────────────────────
add_library(bot_core
  src/coinbase_ws.cpp
  src/log.cpp
)
target_include_directories(bot_core PUBLIC include)
target_link_libraries(bot_core
  PUBLIC
    Boost::headers
    OpenSSL::SSL
    OpenSSL::Crypto
    nlohmann_json::nlohmann_json
    Threads::Threads
)

# ── executable: Phase 1 — Coinbase ticker dump ────────────────────────
add_executable(coinbase_dump src/coinbase_dump_main.cpp)
target_link_libraries(coinbase_dump PRIVATE bot_core)
