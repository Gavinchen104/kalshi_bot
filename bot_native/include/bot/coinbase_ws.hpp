#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <thread>

namespace bot {

struct CoinbaseTick {
    double price;        // BTC-USD spot
    double last_size;    // size of the trade that produced this tick
    std::string time;    // server-provided ISO timestamp (best-effort)
};

using CoinbaseTickHandler = std::function<void(const CoinbaseTick&)>;

class CoinbaseWS {
public:
    CoinbaseWS(std::string host, std::string port, std::string target, std::string product_id);
    ~CoinbaseWS();

    CoinbaseWS(const CoinbaseWS&) = delete;
    CoinbaseWS& operator=(const CoinbaseWS&) = delete;

    void set_handler(CoinbaseTickHandler h);
    void start();   // launches background thread; reconnects on failure
    void stop();    // signals shutdown; joins thread

private:
    void run();

    std::string host_;
    std::string port_;
    std::string target_;
    std::string product_id_;

    CoinbaseTickHandler handler_;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

}  // namespace bot
