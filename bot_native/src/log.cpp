#include "bot/log.hpp"

#include <chrono>
#include <iomanip>

namespace bot {

namespace {
std::mutex& log_mutex() {
    static std::mutex m;
    return m;
}

std::string now_iso() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto secs = time_point_cast<seconds>(now);
    auto ms = duration_cast<milliseconds>(now - secs).count();
    std::time_t tt = system_clock::to_time_t(secs);
    std::tm tm{};
    gmtime_r(&tt, &tm);
    std::ostringstream os;
    os << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S") << '.'
       << std::setw(3) << std::setfill('0') << ms << 'Z';
    return os.str();
}
}  // namespace

void Log::line(std::string_view level, std::string_view event, std::string_view kv) {
    std::lock_guard<std::mutex> g(log_mutex());
    std::cerr << now_iso() << ' ' << level << ' ' << event;
    if (!kv.empty()) std::cerr << ' ' << kv;
    std::cerr << '\n';
}

}  // namespace bot
