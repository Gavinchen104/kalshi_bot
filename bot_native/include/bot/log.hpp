#pragma once

#include <iostream>
#include <mutex>
#include <sstream>
#include <string_view>

namespace bot {

// Simple thread-safe stderr logger. JSON-like key=value format.
// Not allocating anything fancy; this is fine for a hot loop.
class Log {
public:
    static void info(std::string_view event) { line("INFO", event, ""); }
    static void warn(std::string_view event) { line("WARN", event, ""); }
    static void error(std::string_view event) { line("ERROR", event, ""); }

    static void info(std::string_view event, std::string_view kv) { line("INFO", event, kv); }
    static void warn(std::string_view event, std::string_view kv) { line("WARN", event, kv); }
    static void error(std::string_view event, std::string_view kv) { line("ERROR", event, kv); }

private:
    static void line(std::string_view level, std::string_view event, std::string_view kv);
};

// Tiny key=value builder so call sites read cleanly.
struct Kv {
    std::ostringstream os;
    template <typename T>
    Kv& add(std::string_view k, const T& v) {
        if (os.tellp() > 0) os << ' ';
        os << k << '=' << v;
        return *this;
    }
    [[nodiscard]] std::string str() const { return os.str(); }
};

}  // namespace bot
