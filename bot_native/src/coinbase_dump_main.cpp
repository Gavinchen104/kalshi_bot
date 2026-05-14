// Phase 1 smoke binary: subscribes to Coinbase BTC-USD ticker and prints
// one line per tick to stdout. Validates the toolchain + WS pipeline end-to-end.

#include <atomic>
#include <csignal>
#include <cstdio>
#include <thread>

#include "bot/coinbase_ws.hpp"
#include "bot/log.hpp"

namespace {
std::atomic<bool> g_stop{false};
void handle_signal(int) { g_stop.store(true); }
}  // namespace

int main() {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    bot::CoinbaseWS ws("ws-feed.exchange.coinbase.com", "443", "/", "BTC-USD");
    std::atomic<uint64_t> ticks{0};
    ws.set_handler([&](const bot::CoinbaseTick& t) {
        auto n = ticks.fetch_add(1) + 1;
        // Print every tick. For brevity in a real run we'd downsample.
        std::printf("tick=%llu price=%.2f size=%.6f time=%s\n",
                    static_cast<unsigned long long>(n), t.price, t.last_size, t.time.c_str());
        std::fflush(stdout);
    });

    ws.start();
    bot::Log::info("coinbase_dump_started");

    while (!g_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    bot::Log::info("coinbase_dump_stopping");
    ws.stop();
    bot::Log::info("coinbase_dump_done", bot::Kv().add("total_ticks", ticks.load()).str());
    return 0;
}
