// Coinbase Exchange ticker WebSocket client built on Boost.Beast + OpenSSL.
//
// Connects to wss://ws-feed.exchange.coinbase.com, subscribes to the ticker
// channel for BTC-USD, decodes ticker frames as JSON, and emits CoinbaseTick
// values to the registered handler. Reconnects on any error after a short
// backoff.

#include "bot/coinbase_ws.hpp"

#include <boost/asio/connect.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <chrono>
#include <nlohmann/json.hpp>
#include <thread>

#include "bot/log.hpp"

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace asio = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = asio::ip::tcp;
using json = nlohmann::json;

namespace bot {

CoinbaseWS::CoinbaseWS(std::string host, std::string port, std::string target, std::string product_id)
    : host_(std::move(host)), port_(std::move(port)),
      target_(std::move(target)), product_id_(std::move(product_id)) {}

CoinbaseWS::~CoinbaseWS() { stop(); }

void CoinbaseWS::set_handler(CoinbaseTickHandler h) { handler_ = std::move(h); }

void CoinbaseWS::start() {
    bool expected = false;
    if (!running_.compare_exchange_strong(expected, true)) return;
    thread_ = std::thread(&CoinbaseWS::run, this);
}

void CoinbaseWS::stop() {
    running_.store(false);
    if (thread_.joinable()) thread_.join();
}

void CoinbaseWS::run() {
    while (running_.load()) {
        try {
            asio::io_context ioc;
            ssl::context ctx(ssl::context::tlsv12_client);
            ctx.set_default_verify_paths();
            ctx.set_verify_mode(ssl::verify_peer);

            tcp::resolver resolver{ioc};
            websocket::stream<beast::ssl_stream<tcp::socket>> ws{ioc, ctx};

            // SNI is required for Coinbase's TLS endpoint.
            if (!SSL_set_tlsext_host_name(ws.next_layer().native_handle(), host_.c_str())) {
                throw beast::system_error{beast::error_code{static_cast<int>(::ERR_get_error()),
                                                            asio::error::get_ssl_category()},
                                          "SNI host"};
            }

            auto const eps = resolver.resolve(host_, port_);
            asio::connect(beast::get_lowest_layer(ws), eps);

            ws.next_layer().handshake(ssl::stream_base::client);
            ws.handshake(host_, target_);

            // Subscribe to BTC-USD ticker channel.
            json sub = {
                {"type", "subscribe"},
                {"product_ids", {product_id_}},
                {"channels", {"ticker"}},
            };
            ws.write(asio::buffer(sub.dump()));
            Log::info("coinbase_ws_connected", Kv().add("host", host_).add("product", product_id_).str());

            beast::flat_buffer buf;
            while (running_.load()) {
                buf.clear();
                ws.read(buf);
                std::string s = beast::buffers_to_string(buf.data());
                json msg = json::parse(s, nullptr, false);
                if (msg.is_discarded()) continue;
                auto type = msg.value("type", std::string{});
                if (type != "ticker") continue;

                CoinbaseTick t{};
                try {
                    t.price = std::stod(msg.value("price", "0"));
                } catch (...) {
                    continue;
                }
                try {
                    t.last_size = std::stod(msg.value("last_size", "0"));
                } catch (...) {
                    t.last_size = 0.0;
                }
                t.time = msg.value("time", std::string{});
                if (handler_) handler_(t);
            }

            ws.close(websocket::close_code::normal);
        } catch (const std::exception& e) {
            Log::warn("coinbase_ws_reconnect", Kv().add("error", e.what()).str());
        }
        if (!running_.load()) break;
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }
}

}  // namespace bot
