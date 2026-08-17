#include "server-common.h"
#include "http.h"
#include "server-models.h"
#include "server-context.h"
#include "server-stream.h"

#include "build-info.h"
#include "preset.h"
#include "download.h"
#include "http.h"

#include <cpp-httplib/httplib.h> // TODO: remove this once we use HTTP client from download.h
#include <optional>
#include <sheredom/subprocess.h>

#include <functional>
#include <optional>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <queue>
#include <filesystem>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>
#include <cstring>

#ifndef _WIN32
extern char **environ;
#endif

#if defined(__APPLE__) && defined(__MACH__)
// macOS: use _NSGetExecutablePath to get the executable path
#include <mach-o/dyld.h>
#include <limits.h>
#endif

#define DEFAULT_STOP_TIMEOUT 10 // seconds

#define CMD_ROUTER_TO_CHILD_EXIT  "cmd_router_to_child:exit"
#define CMD_CHILD_TO_ROUTER_STATE "cmd_child_to_router:state:" // followed by json string

// address for child process, this is needed because router may run on 0.0.0.0
// ref: https://github.com/ggml-org/llama.cpp/issues/17862
#define CHILD_ADDR "127.0.0.1"

struct server_subproc {
    std::optional<subprocess_s> sproc; // empty while in DOWNLOADING state
    std::atomic<bool> stopped{false}; // set to cancel a download or signal child process exit

    subprocess_s & get() {
        GGML_ASSERT(sproc.has_value() && "subprocess not initialized");
        return sproc.value();
    }

    bool is_alive() {
        return sproc.has_value() && subprocess_alive(&sproc.value());
    }

    void request_exit() {
        if (sproc.has_value()) {
            FILE * stdin_file = subprocess_stdin(&sproc.value());
            if (stdin_file) {
                fprintf(stdin_file, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
                fflush(stdin_file);
            }
        }
        stopped.store(true, std::memory_order_relaxed);
    }

    void terminate() {
        if (!sproc.has_value()) {
            return;
        }
#if defined(_WIN32)
        if (sproc->hProcess == NULL) {
            return;
        }
#else
        if (sproc->child <= 0) {
            return;
        }
#endif
        subprocess_terminate(&sproc.value());
    }
};

// short loopback budget for the resumable stream router to child JSON calls (probe, lookup,
// delete). distinct from params.timeout_read/write which only applies to the generation proxy
static constexpr int STREAM_LOOKUP_TIMEOUT_MS = 250;

static std::filesystem::path get_server_exec_path() {
#if defined(_WIN32)
    wchar_t buf[32768] = { 0 };  // Large buffer to handle long paths
    DWORD len = GetModuleFileNameW(nullptr, buf, _countof(buf));
    if (len == 0 || len >= _countof(buf)) {
        throw std::runtime_error("GetModuleFileNameW failed or path too long");
    }
    return std::filesystem::path(buf);
#elif defined(__APPLE__) && defined(__MACH__)
    char small_path[PATH_MAX];
    uint32_t size = sizeof(small_path);

    if (_NSGetExecutablePath(small_path, &size) == 0) {
        // resolve any symlinks to get absolute path
        try {
            return std::filesystem::canonical(std::filesystem::path(small_path));
        } catch (...) {
            return std::filesystem::path(small_path);
        }
    } else {
        // buffer was too small, allocate required size and call again
        std::vector<char> buf(size);
        if (_NSGetExecutablePath(buf.data(), &size) == 0) {
            try {
                return std::filesystem::canonical(std::filesystem::path(buf.data()));
            } catch (...) {
                return std::filesystem::path(buf.data());
            }
        }
        throw std::runtime_error("_NSGetExecutablePath failed after buffer resize");
    }
#else
    char path[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
    if (count <= 0) {
        throw std::runtime_error("failed to resolve /proc/self/exe");
    }
    return std::filesystem::path(std::string(path, count));
#endif
}

static void unset_reserved_args(common_preset & preset, bool unset_model_args) {
    preset.unset_option("LLAMA_ARG_SSL_KEY_FILE");
    preset.unset_option("LLAMA_ARG_SSL_CERT_FILE");
    preset.unset_option("LLAMA_API_KEY");
    preset.unset_option("LLAMA_ARG_MODELS_DIR");
    preset.unset_option("LLAMA_ARG_MODELS_MAX");
    preset.unset_option("LLAMA_ARG_MODELS_PRESET");
    preset.unset_option("LLAMA_ARG_MODELS_AUTOLOAD");
    preset.unset_option("LLAMA_ARG_MODELS_VRAM_AWARE");
    preset.unset_option("LLAMA_ARG_MODELS_VRAM_MARGIN_MB");
    preset.unset_option("LLAMA_ARG_MODELS_VRAM_OVERHEAD_PCT");
    if (unset_model_args) {
        preset.unset_option("LLAMA_ARG_MODEL");
        preset.unset_option("LLAMA_ARG_MMPROJ");
        preset.unset_option("LLAMA_ARG_ALIAS");
        preset.unset_option("LLAMA_ARG_HF_REPO");
    }
}

#ifdef _WIN32
static std::string wide_to_utf8(const wchar_t * ws) {
    if (!ws || !*ws) {
        return {};
    }

    const int len = static_cast<int>(std::wcslen(ws));
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, ws, len, nullptr, 0, nullptr, nullptr);
    if (bytes == 0) {
        return {};
    }

    std::string utf8(bytes, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws, len, utf8.data(), bytes, nullptr, nullptr);

    return utf8;
}
#endif

static std::vector<std::string> get_environment() {
    std::vector<std::string> env;

#ifdef _WIN32
    LPWCH env_block = GetEnvironmentStringsW();
    if (!env_block) {
        return env;
    }
    for (LPWCH e = env_block; *e; e += wcslen(e) + 1) {
        env.emplace_back(wide_to_utf8(e));
    }
    FreeEnvironmentStringsW(env_block);
#else
    if (environ == nullptr) {
        return env;
    }
    for (char ** e = environ; *e != nullptr; e++) {
        env.emplace_back(*e);
    }
#endif

    return env;
}

void server_model_meta::update_args(common_preset_context & ctx_preset, std::string bin_path) {
    // update params
    unset_reserved_args(preset, false);
    preset.set_option(ctx_preset, "LLAMA_ARG_HOST",  CHILD_ADDR);
    preset.set_option(ctx_preset, "LLAMA_ARG_PORT",  std::to_string(port));
    preset.set_option(ctx_preset, "LLAMA_ARG_ALIAS", name);
    // TODO: maybe validate preset before rendering ?
    // render args
    args = preset.to_args(bin_path);

    // unified binary dispatches by subcommand, re-inject it right after the
    // binary path so the child starts as 'llama serve ...' not 'llama ...'
    const char * app_cmd = std::getenv("LLAMA_APP_CMD");
    if (app_cmd != nullptr && app_cmd[0] != '\0' && !bin_path.empty()) {
        args.insert(args.begin() + 1, app_cmd);
    }
}

void server_model_meta::update_caps() {
    try {
        common_params params;
        preset.apply_to_params(params, {
            "LLAMA_ARG_MODEL",
            "LLAMA_ARG_MODEL_URL",
            "LLAMA_ARG_MMPROJ",
            "LLAMA_ARG_MMPROJ_URL",
            "LLAMA_ARG_MMPROJ_AUTO",
            "LLAMA_ARG_HF_REPO",
            "LLAMA_ARG_HF_REPO_FILE",
        });
        params.offline = true;
        common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
        common_models_handler_apply(handler, params); // note: this won't download the model because offline=true
        if (params.no_mmproj || params.mmproj.path.empty()) {
            multimodal = { false, false };
        } else {
            multimodal = mtmd_get_cap_from_file(params.mmproj.path.c_str());
        }
    } catch (const std::exception & e) {
        LOG_WRN("failed to initialize common_params for multimodal capability detection: %s\n", e.what());
        multimodal = { false, false };
    }
}

//
// server_models
//

server_models::server_models(
        const common_params & params,
        int argc,
        char ** argv)
            : ctx_preset(LLAMA_EXAMPLE_SERVER),
              base_params(params),
              base_env(get_environment()),
              base_preset(ctx_preset.load_from_args(argc, argv)) {
    // clean up base preset
    unset_reserved_args(base_preset, true);
    // set binary path
    try {
        bin_path = get_server_exec_path().string();
    } catch (const std::exception & e) {
        bin_path = argv[0];
        LOG_WRN("failed to get server executable path: %s\n", e.what());
        LOG_WRN("using original argv[0] as fallback: %s\n", argv[0]);
    }
    load_models();
}

void server_models::add_model(server_model_meta && meta) {
    if (mapping.find(meta.name) != mapping.end()) {
        throw std::runtime_error(string_format("model '%s' appears multiple times", meta.name.c_str()));
    }

    // check model name does not conflict with existing aliases
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(meta.name)) {
            throw std::runtime_error(string_format("model name '%s' conflicts with alias of model '%s'",
                meta.name.c_str(), key.c_str()));
        }
    }

    // parse aliases from preset's --alias option (comma-separated)
    std::string alias_str;
    if (meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
        for (auto & alias : string_split<std::string>(alias_str, ',')) {
            alias = string_strip(alias);
            if (!alias.empty()) {
                meta.aliases.insert(alias);
            }
        }
    }

    // parse tags from preset's --tags option (comma-separated)
    std::string tags_str;
    if (meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
        for (auto & tag : string_split<std::string>(tags_str, ',')) {
            tag = string_strip(tag);
            if (!tag.empty()) {
                meta.tags.insert(tag);
            }
        }
    }

    // validate aliases do not conflict with existing names or aliases
    for (const auto & alias : meta.aliases) {
        if (mapping.find(alias) != mapping.end()) {
            throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with existing model name",
                alias.c_str(), meta.name.c_str()));
        }
        for (const auto & [key, inst] : mapping) {
            if (inst.meta.aliases.count(alias)) {
                throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with alias of model '%s'",
                    alias.c_str(), meta.name.c_str(), key.c_str()));
            }
        }
    }

    meta.update_args(ctx_preset, bin_path); // render args
    meta.update_caps();
    std::string name = meta.name;
    mapping[name] = instance_t{
        /* subproc */ std::make_shared<server_subproc>(),
        /* th      */ std::thread(),
        /* meta    */ std::move(meta)
    };
}

void server_models::notify_sse(const std::string & event, const std::string & model_id, const json & data) {
    std::unique_ptr<server_task_result_router> result = std::make_unique<server_task_result_router>();
    result->data = {
        {"model", model_id},
        {"event", event},
    };
    if (!data.is_null()) {
        result->data["data"] = data;
    }
    SRV_DBG("notifying SSE clients about event '%s' for model '%s': %s\n", event.c_str(), model_id.c_str(), safe_json_to_str(result->data).c_str());
    sse.broadcast(std::move(result));
}

void server_models::load_models() {
    // Phase 1: load presets from all sources - pure I/O, no lock needed
    // 1. cached models
    common_presets cached_models = ctx_preset.load_from_cache();
    SRV_INF("Loaded %zu cached model presets\n", cached_models.size());
    // 2. local models from --models-dir
    common_presets local_models;
    if (!base_params.models_dir.empty()) {
        local_models = ctx_preset.load_from_models_dir(base_params.models_dir);
        SRV_INF("Loaded %zu local model presets from %s\n", local_models.size(), base_params.models_dir.c_str());
    }
    // 3. custom-path models from presets
    common_preset global = {};
    common_presets custom_presets = {};
    if (!base_params.models_preset.empty()) {
        custom_presets = ctx_preset.load_from_ini(base_params.models_preset, global);
        SRV_INF("Loaded %zu custom model presets from %s\n", custom_presets.size(), base_params.models_preset.c_str());
    }

    // cascade, apply global preset first
    cached_models  = ctx_preset.cascade(global, cached_models);
    local_models   = ctx_preset.cascade(global, local_models);
    custom_presets = ctx_preset.cascade(global, custom_presets);

    // note: if a model exists in both cached and local, local takes precedence
    common_presets final_presets;
    std::unordered_map<std::string, server_model_source> source_map;
    for (const auto & [name, preset] : cached_models) {
        final_presets[name] = preset;
        source_map[name] = SERVER_MODEL_SOURCE_CACHE;
    }
    for (const auto & [name, preset] : local_models)  {
        final_presets[name] = preset;
        source_map[name] = SERVER_MODEL_SOURCE_MODELS_DIR;
    }
    for (const auto & [name, custom] : custom_presets) {
        if (final_presets.find(name) != final_presets.end()) {
            final_presets[name].merge(custom);
        } else {
            final_presets[name] = custom;
        }
        source_map[name] = SERVER_MODEL_SOURCE_PRESET;
    }

    // overlay router's own CLI args on top of every model preset so that
    // e.g. `llama-server --temp 0` is honoured by all child processes
    for (auto & [name, preset] : final_presets) {
        preset.merge(base_preset);
    }

    auto get_source = [&](const std::string & name) {
        return source_map.count(name) ? source_map.at(name) : SERVER_MODEL_SOURCE_PRESET;
    };

    // Helpers that read `mapping` - must be called while holding the lock.
    std::unordered_set<std::string> custom_names;
    for (const auto & [name, preset] : custom_presets) custom_names.insert(name);
    auto join_set = [](const std::set<std::string> & s) {
        std::string result;
        for (const auto & v : s) {
            if (!result.empty()) result += ", ";
            result += v;
        }
        return result;
    };
    auto log_available_models = [&]() {
        SRV_INF("Available models (%zu) (*: custom preset)\n", mapping.size());
        for (const auto & [name, inst] : mapping) {
            bool has_custom = custom_names.find(name) != custom_names.end();
            std::string info;
            if (!inst.meta.aliases.empty()) info += " (aliases: " + join_set(inst.meta.aliases) + ")";
            if (!inst.meta.tags.empty())    info += " [tags: "    + join_set(inst.meta.tags)    + "]";
            SRV_INF("  %c %s%s\n", has_custom ? '*' : ' ', name.c_str(), info.c_str());
        }
    };
    auto apply_stop_timeout = [&]() {
        for (auto & [name, inst] : mapping) {
            std::string val;
            if (inst.meta.preset.get_option(COMMON_ARG_PRESET_STOP_TIMEOUT, val)) {
                try {
                    inst.meta.stop_timeout = std::stoi(val);
                } catch (...) {
                    SRV_WRN("invalid stop-timeout value '%s' for model '%s', using default %d seconds\n",
                        val.c_str(), name.c_str(), DEFAULT_STOP_TIMEOUT);
                    inst.meta.stop_timeout = DEFAULT_STOP_TIMEOUT;
                }
            }
            // YARD ADDITION (self.ai#128): pin exemption from LRU eviction.
            // Assigned unconditionally, never only-when-present: this lambda also
            // runs on RELOAD, so a `pin` deleted from the preset has to clear the
            // flag. A present-only assignment would leave a model pinned forever
            // with nothing in the preset to explain why.
            std::string pin_val;
            inst.meta.pinned = inst.meta.preset.get_option(COMMON_ARG_PRESET_PIN, pin_val)
                && common_arg_utils::is_truthy(pin_val);
        }
    };
    // update_args() injects HOST/PORT/ALIAS, so strip them before comparing presets
    auto preset_options_for_compare = [](common_preset p) {
        p.unset_option("LLAMA_ARG_HOST");
        p.unset_option("LLAMA_ARG_PORT");
        p.unset_option("LLAMA_ARG_ALIAS");
        return p.options;
    };

    // Phase 2: acquire the lock once for all mapping mutations.
    // We temporarily release it only when calling functions that acquire it internally
    // (unload, load) or when joining threads (the monitoring thread calls update_status
    // which locks the mutex, so joining while holding it would deadlock).
    std::unique_lock<std::mutex> lk(mutex);

    need_reload = false;
    bool is_first_load = mapping.empty();

    if (is_first_load) {
        // FIRST LOAD: add all models, then unlock for autoloading
        for (const auto & [name, preset] : final_presets) {
            server_model_meta meta{
                /* source        */ get_source(name),
                /* preset        */ preset,
                /* name          */ name,
                /* aliases       */ {},
                /* tags          */ {},
                /* port          */ 0,
                /* status        */ SERVER_MODEL_STATUS_UNLOADED,
                /* last_used     */ 0,
                /* args          */ std::vector<std::string>(),
                /* loaded_info   */ {},
                /* progress      */ {},
                /* exit_code     */ 0,
                /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
                /* multimodal    */ mtmd_caps{false, false},
                // /* need_download */ false,
            };
            add_model(std::move(meta));
        }
        apply_stop_timeout();
        log_available_models();

        std::vector<std::string> models_to_load;
        for (const auto & [name, inst] : mapping) {
            std::string val;
            if (inst.meta.preset.get_option(COMMON_ARG_PRESET_LOAD_ON_STARTUP, val) && common_arg_utils::is_truthy(val)) {
                models_to_load.push_back(name);
            }
        }
        if ((int)models_to_load.size() > base_params.models_max) {
            throw std::runtime_error(string_format(
                "number of models to load on startup (%zu) exceeds models_max (%d)",
                models_to_load.size(), base_params.models_max));
        }

        lk.unlock();
        for (const auto & name : models_to_load) {
            SRV_INF("(startup) loading model %s\n", name.c_str());
            load(name);
        }
    } else {
        // RELOAD: diff the new preset list against the current mapping and reconcile
        is_reloading = true;

        // find running models whose source was removed or whose preset changed
        std::vector<std::string> to_unload;
        for (const auto & [name, inst] : mapping) {
            if (!inst.meta.is_running()) continue;
            auto it = final_presets.find(name);
            if (it == final_presets.end()) {
                to_unload.push_back(name); // removed from source
            } else if (preset_options_for_compare(inst.meta.preset) != preset_options_for_compare(it->second)) {
                to_unload.push_back(name); // preset changed
            }
        }

        // unload() acquires the lock internally, so release before each call
        for (const auto & name : to_unload) {
            SRV_INF("(reload) unloading model name=%s (source updated or removed)\n", name.c_str());
            lk.unlock();
            unload(name);
            lk.lock();
        }

        // wait for all targeted models to reach UNLOADED; cv.wait handles unlock/relock
        cv.wait(lk, [&]() {
            for (const auto & name : to_unload) {
                auto it = mapping.find(name);
                if (it != mapping.end() && it->second.meta.is_running()) return false;
            }
            return true;
        });

        // collect all threads to join in one pass while the lock is held:
        // - monitoring threads from just-unloaded models (to_unload)
        // - threads of finished downloads (DOWNLOADED), they acquire the mutex on exit
        // - threads of already-UNLOADED models that are being removed from source
        std::vector<std::thread> threads_to_join;
        for (const auto & name : to_unload) {
            auto it = mapping.find(name);
            if (it != mapping.end() && it->second.th.joinable()) {
                threads_to_join.push_back(std::move(it->second.th));
            }
        }
        for (auto & [name, inst] : mapping) {
            if (inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                continue; // downloading models are not from config sources, leave them alone
            }
            if (inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADED) {
                // joining this thread under the lock deadlocks: it locks the mutex on its way out
                if (inst.th.joinable()) {
                    threads_to_join.push_back(std::move(inst.th));
                }
                continue;
            }
            if (final_presets.find(name) == final_presets.end() && !inst.meta.is_running() && inst.th.joinable()) {
                threads_to_join.push_back(std::move(inst.th));
            }
        }

        // join outside the lock - monitoring thread calls update_status (needs lock)
        lk.unlock();
        for (auto & th : threads_to_join) th.join();
        lk.lock();

        // erase models no longer in any source
        for (auto it = mapping.begin(); it != mapping.end(); ) {
            if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                ++it; // download thread is still busy, skip
            } else if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADED) {
                // download finished, thread is joined above, safe to erase
                GGML_ASSERT(!it->second.th.joinable());
                it = mapping.erase(it);
            } else if (final_presets.find(it->first) == final_presets.end()) {
                SRV_INF("(reload) removing model name=%s (no longer in source)\n", it->first.c_str());
                GGML_ASSERT(!it->second.th.joinable()); // must have been joined above
                it = mapping.erase(it);
            } else {
                ++it;
            }
        }

        // update presets for non-running models still in source
        for (auto & [name, inst] : mapping) {
            if (inst.meta.is_running()) continue;
            auto it = final_presets.find(name);
            if (it == final_presets.end()) continue; // erased above

            inst.meta.preset = it->second;

            // re-parse aliases, then validate against other models
            std::set<std::string> new_aliases;
            std::string alias_str;
            if (inst.meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
                for (auto & alias : string_split<std::string>(alias_str, ',')) {
                    alias = string_strip(alias);
                    if (!alias.empty()) new_aliases.insert(alias);
                }
            }
            inst.meta.aliases.clear();
            for (const auto & alias : new_aliases) {
                bool conflict = false;
                for (const auto & [other_name, other_inst] : mapping) {
                    if (other_name == name) continue;
                    if (other_name == alias || other_inst.meta.aliases.count(alias)) {
                        SRV_WRN("(reload) alias '%s' for model '%s' conflicts with model '%s', skipping\n",
                            alias.c_str(), name.c_str(), other_name.c_str());
                        conflict = true;
                        break;
                    }
                }
                if (!conflict) inst.meta.aliases.insert(alias);
            }

            // re-parse tags
            inst.meta.tags.clear();
            std::string tags_str;
            if (inst.meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
                for (auto & tag : string_split<std::string>(tags_str, ',')) {
                    tag = string_strip(tag);
                    if (!tag.empty()) inst.meta.tags.insert(tag);
                }
            }

            inst.meta.exit_code = 0; // clear failed state so the model can be reloaded
            inst.meta.update_args(ctx_preset, bin_path);
            inst.meta.update_caps();
        }

        // add models that are new in this reload
        std::vector<std::string> newly_added;
        for (const auto & [name, preset] : final_presets) {
            if (mapping.find(name) == mapping.end()) {
                server_model_meta meta{
                    /* source        */ get_source(name),
                    /* preset        */ preset,
                    /* name          */ name,
                    /* aliases       */ {},
                    /* tags          */ {},
                    /* port          */ 0,
                    /* status        */ SERVER_MODEL_STATUS_UNLOADED,
                    /* last_used     */ 0,
                    /* args          */ std::vector<std::string>(),
                    /* loaded_info   */ {},
                    /* progress      */ {},
                    /* exit_code     */ 0,
                    /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
                    /* multimodal    */ mtmd_caps{false, false},
                    // /* need_download */ false,
                };
                add_model(std::move(meta));
                newly_added.push_back(name);
            }
        }

        apply_stop_timeout();

        // clear reload flag before unlocking for autoload - load() blocks on !is_reloading,
        // so clearing it here (while still locked) prevents a deadlock in the autoload calls below
        is_reloading = false;
        cv.notify_all();

        log_available_models();

        // collect autoload candidates while still under the lock
        std::vector<std::string> to_autoload;
        for (const auto & name : newly_added) {
            auto it = mapping.find(name);
            if (it != mapping.end()) {
                std::string val;
                if (it->second.meta.preset.get_option(COMMON_ARG_PRESET_LOAD_ON_STARTUP, val) && common_arg_utils::is_truthy(val)) {
                    to_autoload.push_back(name);
                }
            }
        }

        lk.unlock();
        for (const auto & name : to_autoload) {
            SRV_INF("(reload) loading new model %s\n", name.c_str());
            load(name);
        }

        notify_sse("models_reload", "*");
    }
}

void server_models::update_meta(const std::string & name, const server_model_meta & meta) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        it->second.meta = meta;
    }
    cv.notify_all(); // notify wait_until_loading_finished
}

bool server_models::has_model(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    if (mapping.find(name) != mapping.end()) {
        return true;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return true;
        }
    }
    return false;
}

std::optional<server_model_meta> server_models::get_meta(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    if (need_reload) {
        lk.unlock();
        load_models();
        lk.lock();
    }

    auto it = mapping.find(name);
    if (it != mapping.end()) {
        return it->second.meta;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return inst.meta;
        }
    }
    return std::nullopt;
}

// helper to convert vector<string> to char **
// pointers are only valid as long as the original vector is valid
static std::vector<char *> to_char_ptr_array(const std::vector<std::string> & vec) {
    std::vector<char *> result;
    result.reserve(vec.size() + 1);
    for (const auto & s : vec) {
        result.push_back(const_cast<char*>(s.c_str()));
    }
    result.push_back(nullptr);
    return result;
}

std::vector<server_model_meta> server_models::get_all_meta() {
    std::unique_lock<std::mutex> lk(mutex);
    if (need_reload) {
        lk.unlock();
        load_models();
        lk.lock();
    }

    std::vector<server_model_meta> result;
    result.reserve(mapping.size());
    for (const auto & [name, inst] : mapping) {
        result.push_back(inst.meta);
    }
    return result;
}

// ── VRAM-aware eviction ──────────────────────────────────────────────
//
// Best-effort supplement to the count-based unload_lru() below (issue #22):
// two large models can both fit under models_max but not both fit in VRAM,
// so a load can OOM the child process even while the router still thinks it
// has free "slots". This layer checks free GPU memory before a load and
// proactively evicts LRU resident model(s) to make room, independent of the
// model count.
//
// Two things this deliberately does NOT do, both documented tradeoffs:
//  - It does not link a CUDA (or other backend) memory-info call into the
//    router process. server.cpp is explicit that "router server never loads
//    a model and must not touch the GPU" (it must not create a CUDA primary
//    context) -- see is_router_server / common_params_print_info there. So
//    free VRAM is queried via `nvidia-smi` as an external process instead,
//    matching the existing precedent in api/state.py's _check_gpu().
//  - It does not parse GGUF metadata (param count, quant type) to compute an
//    exact footprint. It uses on-disk file size instead, inflated by a
//    configurable overhead percentage as a coarse proxy for KV-cache /
//    compute-buffer usage. This is an approximation, not a precise figure --
//    real usage also depends on -ngl and context size, neither of which the
//    router has visibility into. It is deliberately biased to over-estimate
//    (a spurious eviction is cheap; an under-estimate that still OOMs is the
//    bug this issue is about).
//
// Both limitations degrade gracefully: if free VRAM can't be queried (no
// nvidia-smi / non-NVIDIA backend) or a model's footprint can't be estimated
// (e.g. a remote HF-repo preset not yet cached locally), eviction for that
// call is skipped and behavior falls back to the pre-existing models_max
// count-based limit only -- never to "evict everything" or "refuse to load".

// Best-effort query of total free VRAM across all visible GPUs, in bytes.
// Returns std::nullopt -- "unknown", never "zero" -- if nvidia-smi isn't on
// PATH, exits non-zero, times out, or its output doesn't parse.
//
// `resident_est_bytes` is the summed estimated footprint of the models
// currently resident on the GPU. The real query ignores it entirely -- the
// driver already accounts for them. It exists only for the fake-total test
// hook below, and callers outside a test build may pass 0.
static std::optional<int64_t> query_free_vram_bytes(int64_t resident_est_bytes) {
    // Test-only overrides, checked before the real nvidia-smi query. None of
    // these is a documented public flag -- real deployments never set them.
    // They exist so tools/server/tests/unit/test_router.py can deterministically
    // exercise VRAM-aware eviction (including forcing multi-model eviction
    // in one load) without real GPU memory pressure, since CI may run these
    // router tests on GPU-less hosts with no nvidia-smi at all.
    //   - LLAMA_TEST_FAKE_TOTAL_VRAM_MB_FILE: path to a text file containing a
    //     fake *total* VRAM capacity in MiB, re-read on every call. Free VRAM
    //     is derived as total - resident_est_bytes, so evicting a model
    //     actually RAISES the next reading, the way a real driver behaves.
    //     The two fixed-value hooks below cannot express that -- they return
    //     whatever the test last wrote, so an eviction never makes room and
    //     evict_for_vram()'s loop can only ever run itself out of candidates
    //     and throw. That is self.llamolotl#34, and this is its fix: eviction
    //     tests need the reading to respond to eviction.
    //   - LLAMA_TEST_FAKE_FREE_VRAM_MB_FILE: path to a text file containing
    //     the fake free-VRAM value in MiB, re-read on every call -- lets a
    //     single already-running test server have "free VRAM" change
    //     mid-test (the real value naturally does too, as models load).
    //   - LLAMA_TEST_FAKE_FREE_VRAM_MB: a static fake value in MiB.
    // Checked most-specific first; a test sets one of them, never several.
    if (const char * fake_total_file = std::getenv("LLAMA_TEST_FAKE_TOTAL_VRAM_MB_FILE")) {
        std::ifstream f(fake_total_file);
        std::string line;
        if (f && std::getline(f, line)) {
            try {
                const int64_t total_bytes = (int64_t) std::stoll(line) * 1024 * 1024;
                // clamp at 0: over-subscribed is "nothing free", not negative free
                return std::max<int64_t>(0, total_bytes - resident_est_bytes);
            } catch (const std::exception &) {
                return std::nullopt; // malformed override -- "unknown", not a real query
            }
        }
        return std::nullopt; // configured but unreadable right now -- same treatment
    }
    if (const char * fake_file = std::getenv("LLAMA_TEST_FAKE_FREE_VRAM_MB_FILE")) {
        std::ifstream f(fake_file);
        std::string line;
        if (f && std::getline(f, line)) {
            try {
                return (int64_t) std::stoll(line) * 1024 * 1024;
            } catch (const std::exception &) {
                return std::nullopt; // malformed override -- "unknown", not a real query
            }
        }
        return std::nullopt; // configured but unreadable right now -- same treatment
    }
    if (const char * fake_mb = std::getenv("LLAMA_TEST_FAKE_FREE_VRAM_MB")) {
        try {
            return (int64_t) std::stoll(fake_mb) * 1024 * 1024;
        } catch (const std::exception &) {
            // fall through to the real nvidia-smi query if this is malformed
        }
    }

    std::vector<std::string> cmd = {
        "nvidia-smi",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    };
    std::vector<char *> argv = to_char_ptr_array(cmd);

    subprocess_s proc;
    int options = subprocess_option_no_window
                | subprocess_option_combined_stdout_stderr
                | subprocess_option_inherit_environment
                | subprocess_option_search_user_path;
    if (subprocess_create(argv.data(), options, &proc) != 0) {
        return std::nullopt;
    }

    std::atomic<bool> done{false};
    std::thread timeout_thread([&]() {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!done.load()) {
            if (std::chrono::steady_clock::now() >= deadline) {
                subprocess_terminate(&proc);
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    std::string output;
    FILE * f = subprocess_stdout(&proc);
    if (f) {
        char buf[256];
        while (fgets(buf, sizeof(buf), f) != nullptr) {
            output += buf;
        }
    }

    done.store(true);
    if (timeout_thread.joinable()) {
        timeout_thread.join();
    }

    int exit_code = 0;
    subprocess_join(&proc, &exit_code);
    subprocess_destroy(&proc);

    if (exit_code != 0 || output.empty()) {
        return std::nullopt;
    }

    // one line per GPU, each a bare number of MiB (nounits)
    int64_t total_free_mib = 0;
    bool parsed_any = false;
    std::istringstream iss(output);
    std::string line;
    while (std::getline(iss, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        try {
            total_free_mib += std::stoll(line);
            parsed_any = true;
        } catch (const std::exception &) {
            // one bad/unexpected line shouldn't necessarily invalidate a
            // query that otherwise parsed fine, e.g. a stray driver warning
            // sharing combined stdout/stderr -- just skip it
        }
    }

    if (!parsed_any) {
        return std::nullopt;
    }
    return total_free_mib * 1024 * 1024;
}

// Estimate a model's resident VRAM footprint in bytes from the size(s) of its
// GGUF file(s) on disk (primary weights + any split shards + mmproj, if
// configured), inflated by `overhead_pct` to approximate KV-cache /
// compute-buffer usage not captured by file size alone. Returns 0 if no
// local file could be resolved (e.g. an HF-repo preset whose weights are not
// in the cache yet) -- callers must treat 0 as "unknown", not "no footprint".
//
// A direct LLAMA_ARG_MODEL only exists for presets that name a local path.
// Router-mode models are normally loaded by HF repo (LLAMA_ARG_HF_REPO), which
// has no path to stat -- so before giving up we resolve the preset the same way
// update_caps() does, with offline=true, which fills in the local cache path
// without downloading anything. Skipping that step is what made
// models_vram_aware silently degrade to count-only eviction for every
// HF-repo-loaded model (#31).
static int64_t cold_start_estimate_bytes(const server_model_meta & meta, int overhead_pct) {
    namespace fs = std::filesystem;

    // GATE (self.llamolotl#36): this heuristic is only allowed where it is an
    // UPPER BOUND on what lands on the card. File size counts every weight in
    // the GGUF, so it is a fair proxy only when nothing is deliberately kept in
    // system RAM. Measured on the live 4090: for fully-offloaded models it sits
    // within 2-11% (Qwen2.5-Coder-32B 22.2 GiB estimated / 22.26 measured,
    // gemma-4 15.9 / 14.48, Qwen2.5-VL 6.7 / 7.48) -- but with expert offload it
    // is 1.8x over for Qwen3-Coder-Next (n-cpu-moe=25) and 6.0x for
    // GLM-4.5-Air-q8_0 (n-cpu-moe=40, 131.3 GiB estimated against 21.33 GiB
    // actually resident), which is what made the router refuse both on a card
    // they fit in with room to spare.
    //
    // So: any option that moves weights off the GPU disqualifies the estimate
    // entirely. Return 0 -- unknown -- and let the load proceed under the
    // child's own --fit, exactly as an unmeasured configuration does.
    //
    // A partial -ngl is NOT disqualifying: it only makes this an over-estimate,
    // and over-estimating is the safe direction here (we may evict more than
    // needed, or refuse something that would have fit; we never OOM). Erring
    // that way is what the old estimator did for every model -- the bug was
    // doing it where the error reached 6x, not the direction itself.
    for (const char * off_gpu : {"LLAMA_ARG_N_CPU_MOE", "LLAMA_ARG_CPU_MOE", "LLAMA_ARG_OVERRIDE_TENSOR"}) {
        std::string unused;
        if (meta.preset.get_option(off_gpu, unused)) {
            return 0;
        }
    }


    auto file_size_with_shards = [](const std::string & path) -> int64_t {
        if (path.empty()) {
            return 0;
        }
        std::error_code ec;
        fs::path p(path);
        if (!fs::exists(p, ec) || ec) {
            return 0;
        }
        int64_t total = (int64_t) fs::file_size(p, ec);
        if (ec) {
            return 0;
        }

        // split GGUF shards look like "<base>-00001-of-00004.gguf"; the router
        // only stores the first shard's path (see
        // common_preset_context::load_from_models_dir), so sum sibling shards
        // in the same directory too.
        static const std::regex re_shard(R"(^(.*)-(\d{5})-of-(\d{5})\.gguf$)");
        std::smatch m;
        std::string filename = p.filename().string();
        if (std::regex_match(filename, m, re_shard) && m[2] == "00001") {
            const std::string base   = m[1].str();
            const std::string suffix = "-of-" + m[3].str() + ".gguf";
            std::error_code dec;
            for (const auto & entry : fs::directory_iterator(p.parent_path(), dec)) {
                if (dec) {
                    break;
                }
                const std::string sibling = entry.path().filename().string();
                if (sibling == filename) {
                    continue; // already counted above
                }
                if (sibling.rfind(base + "-", 0) == 0 && string_ends_with(sibling, suffix)) {
                    std::error_code sec;
                    int64_t ssz = (int64_t) fs::file_size(entry.path(), sec);
                    if (!sec) {
                        total += ssz;
                    }
                }
            }
        }
        return total;
    };

    int64_t total = 0;
    std::string model_path;
    if (meta.preset.get_option("LLAMA_ARG_MODEL", model_path)) {
        total += file_size_with_shards(model_path);
    }
    std::string mmproj_path;
    if (meta.preset.get_option("LLAMA_ARG_MMPROJ", mmproj_path)) {
        total += file_size_with_shards(mmproj_path);
    }

    if (total <= 0) {
        // No local path in the preset. Resolve HF-repo (and *_URL) presets to
        // their cache location. offline=true means this only ever *looks up*
        // where the weights would be -- it never downloads, so a model that has
        // not been fetched yet still correctly reports 0 ("unknown").
        try {
            common_params params;
            meta.preset.apply_to_params(params, {
                "LLAMA_ARG_MODEL",
                "LLAMA_ARG_MODEL_URL",
                "LLAMA_ARG_MMPROJ",
                "LLAMA_ARG_MMPROJ_URL",
                "LLAMA_ARG_MMPROJ_AUTO",
                "LLAMA_ARG_HF_REPO",
                "LLAMA_ARG_HF_REPO_FILE",
            });
            params.offline = true;

            common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
            common_models_handler_apply(handler, params);

            total += file_size_with_shards(params.model.path);
            if (!params.no_mmproj) {
                total += file_size_with_shards(params.mmproj.path);
            }
        } catch (const std::exception & e) {
            // Same posture as update_caps(): a preset we cannot resolve is not
            // fatal here. Returning 0 below keeps the safe "unknown, don't
            // guess" path that evict_for_vram() already handles.
            LOG_WRN("failed to resolve model path for VRAM footprint estimate (model=%s): %s\n",
                    meta.name.c_str(), e.what());
        }
    }

    if (total <= 0) {
        return 0; // unresolved local path (e.g. not downloaded yet) -- unknown, not zero footprint
    }

    return total + (total * (int64_t) std::max(0, overhead_pct)) / 100;
}

// An operator-declared footprint for this configuration, in bytes, or 0 if the
// preset does not carry one (issue #39).
//
// This is NOT a relaxation of "never guess a footprint". A declared value is a
// measurement -- it is what somebody read off this card for this configuration
// and then wrote down -- whereas cold_start_estimate_bytes() derives a number
// from a file size, which is why that one has to disqualify itself the moment
// weights are deliberately kept in system RAM. The distinction that matters is
// where the number came from, not whether it is a literal.
//
// It exists because that disqualification leaves split (-ncmoe / -ot)
// configurations with no admission check at all until the router has loaded
// them once. Observed in production: GLM-4.5-Air-q8_0 (n-cpu-moe=40, 21.3 GiB
// measured) was admitted onto a card with 19.6 GiB free because another tenant
// held 4.4 GiB, and died on CUDA OOM. The child's own --fit is no help there --
// n-cpu-moe and n-gpu-layers are both pinned in the preset, so fitting has no
// layer budget left to reduce.
//
// Declaring a value is the operator saying "I measured this, use it until you
// have measured it yourself". A real measurement always outranks it.
static int64_t declared_footprint_bytes(const server_model_meta & meta) {
    std::string value;
    if (!meta.preset.get_option(COMMON_ARG_PRESET_VRAM_FOOTPRINT_MIB, value)) {
        return 0;
    }
    try {
        const int64_t mib = std::stoll(value);
        if (mib <= 0) {
            return 0; // 0 or negative means "no claim", same as absent
        }
        return mib * 1024 * 1024;
    } catch (const std::exception &) {
        LOG_WRN("ignoring unparseable vram-footprint-mib for model=%s: '%s'\n", meta.name.c_str(), value.c_str());
        return 0;
    }
}

// How much VRAM one expert layer frees when moved to system RAM (issue #29).
//
// Zero means "no claim", and a model with no claim is refused when it does not
// fit, exactly as before. This is deliberately operator-declared rather than
// derived: the cost varies enormously by model (~1298 MiB/layer measured on
// GLM-4.5-Air against ~664 MiB/layer on Qwen3-Coder-Next) and the router has no
// architecture metadata to compute it from. Both figures were already measured
// and written down in the deployment's preset comments; this only makes them
// machine-readable.
static int64_t shed_mib_per_layer(const server_model_meta & meta) {
    std::string value;
    if (!meta.preset.get_option(COMMON_ARG_PRESET_VRAM_SHED_MIB_PER_LAYER, value)) {
        return 0;
    }
    try {
        const int64_t mib = std::stoll(value);
        return mib > 0 ? mib : 0;
    } catch (const std::exception &) {
        LOG_WRN("ignoring unparseable vram-shed-mib-per-layer for model=%s: '%s'\n",
                meta.name.c_str(), value.c_str());
        return 0;
    }
}

// The model's current expert-offload setting. Absent means 0 -- nothing on the
// CPU yet, which is a legitimate starting point to shed from.
static int current_n_cpu_moe(const server_model_meta & meta) {
    std::string value;
    if (!meta.preset.get_option("LLAMA_ARG_N_CPU_MOE", value)) {
        return 0;
    }
    try {
        const int n = std::stoi(value);
        return n > 0 ? n : 0;
    } catch (const std::exception &) {
        return 0;
    }
}

// Identity of a model CONFIGURATION for footprint purposes (issue #36).
//
// The effective preset with the per-instance injections update_args() adds
// (HOST/PORT/ALIAS) stripped back out, rendered to INI. Two launches of the same
// configuration produce the same key; changing n-cpu-moe, n-gpu-layers,
// ctx-size or a cache type produces a different one, which is the entire point
// -- those are exactly the options that move the footprint.
//
// vram-footprint-mib is stripped for the opposite reason: it is a claim ABOUT
// the configuration, not part of it. Leaving it in would mean declaring a value
// invalidates every measurement already taken for that same configuration.
std::string server_models::footprint_key(const server_model_meta & meta) {
    common_preset p = meta.preset;
    p.unset_option("LLAMA_ARG_HOST");
    p.unset_option("LLAMA_ARG_PORT");
    p.unset_option("LLAMA_ARG_ALIAS");
    p.unset_option(COMMON_ARG_PRESET_VRAM_FOOTPRINT_MIB);
    // Also a claim ABOUT a configuration rather than part of one (issue #29):
    // two otherwise-identical presets that disagree only on the per-layer shed
    // cost describe the same child process and must share its measurement.
    p.unset_option(COMMON_ARG_PRESET_VRAM_SHED_MIB_PER_LAYER);
    // And pin (self.ai#128), for a third time the same reason: it is router
    // eviction POLICY, not part of the child's configuration. Two presets that
    // differ only in pin launch byte-identical children and must share the
    // measurement -- otherwise pinning a model throws away every footprint
    // already measured for it, which is the opposite of what pinning is for.
    p.unset_option(COMMON_ARG_PRESET_PIN);
    return p.to_ini();
}

// Record what a child measured about itself, from the ready payload it sends
// (see get_model_info(), tools/server/server-context.cpp). The child reports a
// per-device breakdown; what matters here is how much of the CARD a load of
// this configuration costs, so device entries are summed and host memory is
// excluded by the child already -- weights that -ncmoe pushed to system RAM are
// precisely what must not count against the card.
//
// Caller must hold mutex.
void server_models::record_measured_footprint(const server_model_meta & meta) {
    if (!meta.loaded_info.is_object()) {
        return;
    }
    auto it_mem = meta.loaded_info.find("memory");
    if (it_mem == meta.loaded_info.end() || !it_mem->is_array()) {
        return; // older child build, or a wake-from-sleep with no payload
    }

    int64_t total = 0;
    for (const auto & dev : *it_mem) {
        if (!dev.is_object()) {
            continue;
        }
        total += (int64_t) json_value(dev, "model",   (uint64_t) 0);
        total += (int64_t) json_value(dev, "context", (uint64_t) 0);
        total += (int64_t) json_value(dev, "compute", (uint64_t) 0);
    }
    if (total <= 0) {
        // Test-only override, in the same family as the LLAMA_TEST_FAKE_*_VRAM_MB
        // hooks in query_free_vram_bytes() and never set in a real deployment.
        //
        // CI runs these tests against a stub CUDA driver, so a child reports
        // "memory": [] -- there are no devices to attribute anything to. Every
        // measurement is therefore 0 there, which makes any test of "a
        // measurement outranks X" unsatisfiable rather than merely red. This
        // supplies the figure the child would have reported on a real card.
        //
        // Deliberately placed AFTER the payload sum, not before it: a genuine
        // measurement always wins, so the hook cannot mask a real reading.
        if (const char * fake_mib = std::getenv("LLAMA_TEST_FAKE_MEASURED_MIB")) {
            try {
                total = (int64_t) std::stoll(fake_mib) * 1024 * 1024;
            } catch (const std::exception &) {
                return;
            }
        }
    }
    if (total <= 0) {
        return; // a CPU-only load holds no card; nothing to remember
    }

    const std::string key = footprint_key(meta);
    measured_footprints[key] = total;
    SRV_INF("measured footprint for name=%s: %lld MiB\n", meta.name.c_str(), (long long) (total / (1024 * 1024)));
}

// Caller must hold mutex.
std::optional<int64_t> server_models::lookup_measured_footprint(const server_model_meta & meta) {
    auto it = measured_footprints.find(footprint_key(meta));
    if (it == measured_footprints.end()) {
        return std::nullopt;
    }
    return it->second;
}

// The one place that decides how big a configuration is, in bytes, or 0 for
// "unknown". Every caller in evict_for_vram() goes through here so the incoming
// model and the resident models can never be sized by different rules -- sizing
// an arrival generously and the residents meanly is how an admission check
// starts refusing loads that fit.
//
// Precedence, most trustworthy first:
//   1. measured  -- what a child of ours reported after loading this exact
//                   configuration in this process
//   2. declared  -- vram-footprint-mib in the preset: a measurement somebody
//                   took and wrote down (#39)
//   3. estimated -- cold_start_estimate_bytes(): GGUF size + overhead, and only
//                   where that is an upper bound (it disqualifies itself for
//                   -ncmoe / -ot, which is what makes (2) necessary at all)
//   4. unknown   -- 0. Not a size. evict_for_vram() steps aside and the load
//                   proceeds under models_max and the child's own --fit.
//
// Caller must hold mutex.
footprint_info server_models::footprint_bytes(const server_model_meta & meta) {
    if (auto measured = lookup_measured_footprint(meta)) {
        return {*measured, "measured"};
    }
    if (const int64_t declared = declared_footprint_bytes(meta); declared > 0) {
        return {declared, "declared"};
    }
    const int64_t estimated = cold_start_estimate_bytes(meta, base_params.models_vram_overhead_pct);
    if (estimated > 0) {
        return {estimated, "estimated"};
    }
    return {}; // unknown: bytes 0, source nullptr -- absence of information, not "free"
}

// Public, lock-taking wrapper for reporting (self.ai#107). GET /v1/models runs
// outside the mutex over copies from get_all_meta(), but measured_footprints is
// shared state, so the lookup must take the lock itself.
footprint_info server_models::footprint_report(const server_model_meta & meta) {
    std::unique_lock<std::mutex> lk(mutex);
    return footprint_bytes(meta);
}

void server_models::evict_for_vram(const server_model_meta & incoming) {
    if (!base_params.models_vram_aware) {
        return;
    }

    // How big this configuration is: measured, else declared, else estimated,
    // else unknown. See footprint_bytes() for why the order is that order.
    //
    // Unknown is 0, and 0 means we step aside: fall back to the models_max count
    // limit and let the load proceed under the child's own parameter fitting
    // (--fit, on by default with a per-device target margin). Refusing on a
    // number we do not have is how issue #36 got filed -- sizing a model by its
    // GGUF file x an overhead percentage put GLM-4.5-Air-q8_0 at 131 GiB when it
    // occupies 21, and refused it on a card it fits in with room to spare.
    //
    // Stepping aside is not free either, and #39 is the other edge of it: --fit
    // cannot rescue a configuration that pins both n-cpu-moe and n-gpu-layers,
    // because there is no layer budget left for it to reduce. That is what
    // vram-footprint-mib is for.
    int64_t incoming_bytes = 0;
    {
        std::unique_lock<std::mutex> lk(mutex);
        incoming_bytes = footprint_bytes(incoming).bytes;
    }
    if (incoming_bytes <= 0) {
        return;
    }
    // The margin now carries a second job worth naming: a measured footprint is
    // llama.cpp's own accounting, which excludes the CUDA primary context the
    // incoming child will create (a few hundred MiB the driver reports but
    // llama.cpp never allocates). The default 512 MiB margin covers it.
    const int64_t margin_bytes = (int64_t) base_params.models_vram_margin_mb * 1024 * 1024;

    // bounded: each iteration evicts exactly one resident model, so this always terminates
    for (;;) {
        // resident_est_bytes is ignored by the real nvidia-smi path (the driver
        // already accounts for resident models); it exists so the fake-total
        // test hook can derive free VRAM as total - resident, which is what
        // makes an eviction actually raise the next reading (#34).
        int64_t resident_bytes = 0;
        {
            std::unique_lock<std::mutex> lk(mutex);
            for (const auto & [key, inst] : mapping) {
                if (inst.meta.status != SERVER_MODEL_STATUS_LOADED) {
                    continue;
                }
                resident_bytes += footprint_bytes(inst.meta).bytes;
            }
        }
        auto free_bytes = query_free_vram_bytes(resident_bytes);
        if (!free_bytes.has_value()) {
            // unknown free VRAM (no nvidia-smi / non-NVIDIA backend) -- never treat that as
            // "zero free", which would evict everything; just fall back to models_max only
            return;
        }
        if (incoming_bytes + margin_bytes <= *free_bytes) {
            return; // already enough room
        }

        // YARD EDIT (self.ai#128): pin is a PREFERENCE here, not a veto.
        //
        // unload_lru() treats a pin as absolute, because there the alternative is
        // only "one more model than models_max wanted" -- a bookkeeping limit.
        // Here the alternative is a real OOM or a refused load, so an absolute pin
        // would mean a big model becomes permanently unloadable the moment enough
        // support models are pinned. Measured on the deployed 4090: the two pinned
        // support models hold 1308 MiB (846 + 462) even at n-gpu-layers=0 -- the
        // CUDA primary context plus KV/compute buffers -- and the largest model
        // needs 22784 + 512 margin against a 24564 MiB card. An absolute pin there
        // refuses it outright.
        //
        // So: take the LRU UNPINNED model whenever one exists, and only fall back
        // to a pinned one when nothing else is left. Support models then survive
        // every ordinary swap (which is what a pin is for) while the biggest model
        // still loads when it genuinely needs the whole card. The loop evicts one
        // per iteration, so the fallback also stays minimal -- it gives up exactly
        // as many pinned models as the fit requires, not all of them.
        std::string lru_name;
        int64_t lru_last_used = ggml_time_ms();
        std::string lru_pinned_name;
        int64_t lru_pinned_last_used = ggml_time_ms();
        {
            std::unique_lock<std::mutex> lk(mutex);
            for (const auto & [key, inst] : mapping) {
                // only LOADED models actually hold GPU memory: a LOADING model is mid-request
                // (evicting it is exactly the "still in active use" case we must not do), and a
                // SLEEPING model has already released its weights (destroy() runs on entering
                // sleep, see server_context_impl::handle_sleeping_state), so evicting it wouldn't
                // free anything anyway.
                if (inst.meta.status != SERVER_MODEL_STATUS_LOADED) {
                    continue;
                }
                if (inst.meta.pinned) {
                    if (lru_pinned_name.empty() || inst.meta.last_used < lru_pinned_last_used) {
                        lru_pinned_name = key;
                        lru_pinned_last_used = inst.meta.last_used;
                    }
                    continue;
                }
                if (lru_name.empty() || inst.meta.last_used < lru_last_used) {
                    lru_name = key;
                    lru_last_used = inst.meta.last_used;
                }
            }
        }

        bool overriding_pin = false;
        if (lru_name.empty() && !lru_pinned_name.empty()) {
            // Ordered BEFORE the expert-layer shed below, deliberately. The shed
            // degrades the INCOMING model for its whole residency -- and the
            // incoming model is the hot path, the thing a user waits on token by
            // token. Overriding a pin costs one cold reload of a support model the
            // next time something reranks or embeds. On a card dedicated to
            // serving, a permanently slower chat model is the worse trade.
            //
            // This also keeps #29's stated invariant intact: the shed is still the
            // last thing tried before refusing, and still "competes with a
            // refusal, never with a cheaper option" -- overriding a pin is simply
            // one of the cheaper options that now comes first.
            //
            // Last resort. WARN rather than INFO: a pin was asked for and is not
            // being honoured, and the operator's recourse (shrink the pinned set,
            // or the incoming model) is only findable if this is visible.
            SRV_WRN("vram-aware eviction: no unpinned model left to evict; overriding the pin on "
                    "name=%s to fit name=%s\n", lru_pinned_name.c_str(), incoming.name.c_str());
            lru_name = lru_pinned_name;
            overriding_pin = true;
        }

        if (lru_name.empty()) {
            // nothing left we're allowed to evict and the incoming model still doesn't fit --
            // this is the model-doesn't-fit-at-all case (not the two-models-that-should-have-been-
            // evicted case #22 handles). Don't let the load proceed into a real OOM: raise the
            // specific terminal error (issue #27), which propagates cleanly out of load() -- we
            // are called before load() takes its own lock and before any child instance exists,
            // and the LRU-scan lock above is already released here, so no lock is held. ex_wrapper
            // maps this to a structured HTTP 503 (T-006).
            // Issue #29: before refusing, try to make the model FIT by pushing more
            // expert layers to system RAM. "Slower but serving" beats "gone" --
            // shedding is the whole point of having an MoE offload knob.
            //
            // Only reachable when nothing else can be evicted, so this competes with
            // a refusal, never with a cheaper option.
            // Budget is free MINUS the margin, mirroring the fit test above
            // (incoming + margin <= free). Passing free + margin would under-shed
            // by twice the margin and hand the child a config that still OOMs.
            if (try_shed_to_fit(incoming, incoming_bytes, *free_bytes - margin_bytes)) {
                return; // respawn will use the shed config; the load proceeds
            }
            throw server_model_vram_unfittable_error(incoming.name, incoming_bytes, *free_bytes);
        }

        SRV_INF("vram-aware eviction: %lld MiB free < needed %lld MiB (est. %lld MiB + %d MiB margin), "
                "evicting LRU model name=%s%s\n",
                (long long) (*free_bytes / (1024 * 1024)),
                (long long) ((incoming_bytes + margin_bytes) / (1024 * 1024)),
                (long long) (incoming_bytes / (1024 * 1024)),
                base_params.models_vram_margin_mb,
                lru_name.c_str(),
                overriding_pin ? " (PINNED -- no unpinned candidate left)" : "");

        unload(lru_name);
        // wait for unload to complete
        {
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk, [this, &lru_name]() {
                return mapping[lru_name].meta.status == SERVER_MODEL_STATUS_UNLOADED;
            });
        }
        // loop: re-query free VRAM fresh from the driver (ground truth, not our estimate),
        // which naturally reflects however much the evicted model actually freed
    }
}

// Push more expert layers to system RAM so a model that does not fit, fits
// (issue #29). Returns true when the model's config was rewritten and the load
// should proceed; false when we cannot size a shed and the caller should refuse.
//
// The lever is -ncmoe, which is applied strictly at LOAD time -- the tensor's
// buffer is chosen when it is created and there is no ggml API to move an
// allocated tensor between backends. So this is not a "warm rebalance" of a
// resident model: it rewrites the config of a model that is about to spawn.
// That is exactly why it is cheap (pure orchestration, no llama-internals
// surgery) and why it is only useful here, on the not-yet-loaded path.
//
// Sizing is operator-declared via vram-shed-mib-per-layer. A linear model is
// crude, but it is anchored on a measured point and only has to get us over a
// threshold the child then re-checks for itself -- llama-server still refuses
// or OOMs on its own terms if this lands short, so an optimistic estimate
// cannot turn a refusal into silent corruption.
bool server_models::try_shed_to_fit(const server_model_meta & incoming,
                                    int64_t needed_bytes,
                                    int64_t budget_bytes) {
    const int64_t per_layer_mib = shed_mib_per_layer(incoming);
    if (per_layer_mib <= 0) {
        return false; // no claim -> refuse, exactly as before #29
    }
    const int64_t per_layer_bytes = per_layer_mib * 1024 * 1024;

    const int64_t deficit = needed_bytes - budget_bytes;
    if (deficit <= 0) {
        return false; // not the shortfall case; nothing to do
    }

    // Round UP: shedding half a layer is not a thing, and landing one layer
    // heavy is an OOM while landing one layer light is only slower.
    const int64_t extra_layers = (deficit + per_layer_bytes - 1) / per_layer_bytes;
    const int current = current_n_cpu_moe(incoming);
    const int64_t shed_bytes = extra_layers * per_layer_bytes;

    // The declaration must move with the config it describes. Leaving the old
    // figure on a shed model would re-refuse it on the next load for a size it
    // no longer has. If shedding would take the declared footprint to zero the
    // linear model has been extrapolated past where it means anything -- refuse
    // rather than spawn a child on a number we do not believe.
    const int64_t declared = declared_footprint_bytes(incoming);
    if (declared > 0 && declared - shed_bytes <= 0) {
        SRV_INF("vram shed: model name=%s needs %lld more layers on CPU than its declared "
                "footprint can account for; refusing rather than guessing\n",
                incoming.name.c_str(), (long long) extra_layers);
        return false;
    }

    const int new_ncmoe = current + (int) extra_layers;

    SRV_INF("vram shed: model name=%s does not fit (needs %lld MiB, budget %lld MiB); "
            "moving %lld more expert layer(s) to system RAM (-ncmoe %d -> %d, "
            "%lld MiB/layer). SLOWER BUT SERVING -- this is issue #29's degrade path, "
            "not a measurement\n",
            incoming.name.c_str(),
            (long long) (needed_bytes / (1024 * 1024)),
            (long long) (budget_bytes / (1024 * 1024)),
            (long long) extra_layers, current, new_ncmoe,
            (long long) per_layer_mib);

    {
        std::unique_lock<std::mutex> lk(mutex);
        auto it = mapping.find(incoming.name);
        if (it == mapping.end()) {
            return false;
        }
        auto & meta = it->second.meta;
        meta.preset.set_option(ctx_preset, "LLAMA_ARG_N_CPU_MOE", std::to_string(new_ncmoe));
        if (declared > 0) {
            meta.preset.set_option(ctx_preset, COMMON_ARG_PRESET_VRAM_FOOTPRINT_MIB,
                                   std::to_string((declared - shed_bytes) / (1024 * 1024)));
        }
        // Re-render argv so the child actually spawns with the shed setting. Without
        // this the preset says one thing and the process another.
        meta.update_args(ctx_preset, bin_path);
    }
    return true;
}

void server_models::unload_lru() {
    if (base_params.models_max <= 0) {
        return; // no limit
    }
    // remove one of the servers if we passed the models_max (least recently used - LRU)
    std::string lru_model_name = "";
    int64_t lru_last_used = ggml_time_ms();
    size_t count_active = 0;
    size_t count_pinned = 0;
    {
        std::unique_lock<std::mutex> lk(mutex);
        for (const auto & m : mapping) {
            if (m.second.meta.is_running()) {
                count_active++;
                // YARD EDIT (self.ai#128): a pinned model is counted but never
                // chosen. Support models -- the embedder, the reranker -- are by
                // nature the least recently used things running, because nothing
                // touches them between requests. Plain LRU therefore evicts
                // exactly the models that must stay resident, and the eviction is
                // silent: the next embedding request simply reloads and pays the
                // latency. Counting them still is the whole point of a pin -- it
                // reserves a slot rather than conjuring one, so the VRAM
                // arithmetic and models_max keep meaning what they say.
                if (m.second.meta.pinned) {
                    count_pinned++;
                    continue;
                }
                if (m.second.meta.last_used < lru_last_used) {
                    lru_model_name = m.first;
                    lru_last_used = m.second.meta.last_used;
                }
            }
        }
    }
    if (lru_model_name.empty() && count_active >= (size_t)base_params.models_max) {
        // Every running model is pinned, so there is nothing this function may
        // evict. Say so: load() will throw "model limit reached" a moment later,
        // and without this line the cause -- an over-pinned preset, not a busy
        // router -- is invisible.
        SRV_WRN("models_max (%d) reached and all %zu running models are pinned; "
            "nothing is evictable\n", base_params.models_max, count_pinned);
    }
    if (!lru_model_name.empty() && count_active >= (size_t)base_params.models_max) {
        SRV_INF("models_max limit reached, removing LRU name=%s (%zu pinned, exempt)\n",
            lru_model_name.c_str(), count_pinned);
        unload(lru_model_name);
        // wait for unload to complete
        {
            std::unique_lock<std::mutex> lk(mutex);
            cv.wait(lk, [this, &lru_model_name]() {
                return mapping[lru_model_name].meta.status == SERVER_MODEL_STATUS_UNLOADED;
            });
        }
    }
}

void server_models::load(const std::string & name) {
    load(name, load_options{});
}

void server_models::load(const std::string & name, const load_options & opts) {
    if (!opts.custom_meta.has_value()) {
        if (!has_model(name)) {
            throw std::runtime_error("model name=" + name + " is not found");
        }
        unload_lru(); // count-based limit (models_max)
        auto incoming_meta = get_meta(name);
        if (incoming_meta.has_value()) {
            evict_for_vram(*incoming_meta); // VRAM-aware limit, layered on top (issue #22)
        }
    }

    std::unique_lock<std::mutex> lk(mutex);
    // edge case: block until any in-progress reload has finished so we always load
    // against the freshest preset and a consistent mapping state
    cv.wait(lk, [this]() { return !is_reloading; });

    auto meta = opts.custom_meta.has_value() ? *opts.custom_meta : mapping[name].meta;
    if (meta.status != SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model %s is not ready\n", name.c_str());
        return;
    }

    // Re-check capacity under the lock to prevent concurrent loads from
    // exceeding models_max. Without this, the window between unload_lru()
    // releasing its lock and this lock_guard acquiring allows multiple
    // threads to each observe capacity and all proceed to load.
    if (base_params.models_max > 0) {
        size_t count_active = 0;
        for (const auto & m : mapping) {
            if (m.second.meta.is_running()) {
                count_active++;
            }
        }
        if (count_active >= (size_t)base_params.models_max) {
            throw std::runtime_error("model limit reached, try again later");
        }
    }

    // prepare new instance info
    instance_t inst;
    inst.meta             = meta;
    inst.meta.port        = common_http_get_free_port();
    inst.meta.status      = SERVER_MODEL_STATUS_LOADING;
    inst.meta.loaded_info = json{};
    inst.meta.last_used   = ggml_time_ms();

    if (inst.meta.port <= 0) {
        throw std::runtime_error("failed to get a port number");
    }

    inst.subproc = std::make_shared<server_subproc>();
    {
        SRV_INF("spawning server instance with name=%s on port %d\n", inst.meta.name.c_str(), inst.meta.port);

        inst.meta.update_args(ctx_preset, bin_path); // render args

        std::vector<std::string> child_args = inst.meta.args; // copy
        std::vector<std::string> child_env  = base_env; // copy
        child_env.push_back("LLAMA_SERVER_ROUTER_PORT=" + std::to_string(base_params.port));

        if (opts.mode == SERVER_CHILD_MODE_DOWNLOAD) {
            inst.meta.status = SERVER_MODEL_STATUS_DOWNLOADING;
            child_env.push_back("LLAMA_SERVER_CHILD_MODE=download");
            child_env.push_back("LLAMA_ARG_HF_REPO=" + name);
        }

        SRV_INF("%s", "spawning server instance with args:\n");
        for (const auto & arg : child_args) {
            SRV_INF("  %s\n", arg.c_str());
        }
        inst.meta.args = child_args; // save for debugging

        std::vector<char *> argv = to_char_ptr_array(child_args);
        std::vector<char *> envp = to_char_ptr_array(child_env);

        // TODO @ngxson : maybe separate stdout and stderr in the future
        //                so that we can use stdout for commands and stderr for logging
        int options = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
        inst.subproc->sproc.emplace();
        int result = subprocess_create_ex(argv.data(), options, envp.data(), &inst.subproc->get());
        if (result != 0) {
            throw std::runtime_error("failed to spawn server instance");
        }
    }

    // start a thread to manage the child process
    // captured variables are guaranteed to be destroyed only after the thread is joined
    inst.th = std::thread([
        this, name,
        child_proc = inst.subproc,
        port = inst.meta.port,
        stop_timeout = inst.meta.stop_timeout,
        child_mode = opts.mode
    ]() {
        FILE * stdin_file = subprocess_stdin(&child_proc->get());
        FILE * stdout_file = subprocess_stdout(&child_proc->get()); // combined stdout/stderr

        std::thread log_thread([&]() {
            // read stdout/stderr and forward to main server log
            // also handle status report from child process
            std::vector<char> vec_buf(128 * 1024); // large buffer for storing info
            char * buffer = vec_buf.data();
            if (stdout_file) {
                while (fgets(buffer, vec_buf.size(), stdout_file) != nullptr) {
                    LOG("[%5d] %s", port, buffer);
                    std::string str(buffer);
                    if (string_starts_with(buffer, CMD_CHILD_TO_ROUTER_STATE)) {
                        this->handle_child_state(name, str);
                    }
                }
            } else {
                SRV_ERR("failed to get stdout/stderr of child process for name=%s\n", name.c_str());
            }
        });

        std::thread stopping_thread([&]() {
            // thread to monitor explicit stop requests; child crash is signalled via child_proc->stopped
            auto is_stopping = [this, &name]() {
                return this->stopping_models.find(name) != this->stopping_models.end();
            };
            {
                std::unique_lock<std::mutex> lk(this->mutex);
                this->cv_stop.wait(lk, [&]() {
                    return is_stopping() || child_proc->stopped.load(std::memory_order_acquire);
                });
            }
            // child crashed or finished on its own, skip graceful shutdown sequence
            if (child_proc->stopped.load(std::memory_order_acquire)) {
                return;
            }
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            fprintf(stdin_file, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
            fflush(stdin_file);
            int64_t start_time = ggml_time_ms();
            while (true) {
                std::unique_lock<std::mutex> lk(this->mutex);
                if (!is_stopping() || child_proc->stopped.load(std::memory_order_acquire)) {
                    return;
                }
                int64_t elapsed = ggml_time_ms() - start_time;
                if (elapsed >= stop_timeout * 1000) {
                    lk.unlock();
                    SRV_WRN("force-killing model instance name=%s after %d seconds timeout\n", name.c_str(), stop_timeout);
                    child_proc->terminate();
                    return;
                }
                this->cv_stop.wait_for(lk, std::chrono::seconds(1), [&]() {
                    return !is_stopping() || child_proc->stopped.load(std::memory_order_acquire);
                });
            }
        });

        // we reach here when the child process exits (stdout EOF)
        // note: we cannot join() prior to this point because it will close stdin_file
        if (log_thread.joinable()) {
            log_thread.join();
        }

        child_proc->stopped.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(this->mutex);
            stopping_models.erase(name);
            cv_stop.notify_all();
        }
        if (stopping_thread.joinable()) {
            stopping_thread.join();
        }

        // get the exit code
        int exit_code = 0;
        subprocess_join(&child_proc->get(), &exit_code);
        subprocess_destroy(&child_proc->get());

        // update status and exit code
        if (child_mode == SERVER_CHILD_MODE_DOWNLOAD) {
            // instance will be cleaned up on next load_models() call
        } else {
            this->update_status(name, {
                SERVER_MODEL_STATUS_UNLOADED,
                exit_code
            });
        }
        SRV_INF("instance name=%s exited with status %d\n", name.c_str(), exit_code);
    });

    // clean up old process/thread if exists
    {
        auto & old_instance = mapping[name];
        // old process should have exited already, but just in case, we clean it up here
        if (old_instance.subproc && old_instance.subproc->is_alive()) {
            SRV_WRN("old process for model name=%s is still alive, this is unexpected\n", name.c_str());
            old_instance.subproc->terminate(); // force kill
        }
        if (old_instance.th.joinable()) {
            old_instance.th.join();
        }
    }

    notify_sse("model_status", name, {
        {"status", server_model_status_to_string(inst.meta.status)},
    });

    mapping[name] = std::move(inst);
    cv.notify_all();
}

void server_models::unload(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
            SRV_INF("cancelling download for model name=%s\n", name.c_str());
            it->second.subproc->request_exit();
            // for convenience, we wait the status change here
            wait(lk, name, [](const server_model_meta & new_meta) {
                return new_meta.status != SERVER_MODEL_STATUS_DOWNLOADING;
            });
        } else if (it->second.meta.is_running()) {
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            stopping_models.insert(name);
            if (it->second.meta.status == SERVER_MODEL_STATUS_LOADING) {
                // special case: if model is in loading state, unloading means force-killing it
                SRV_WRN("model name=%s is still loading, force-killing\n", name.c_str());
                it->second.subproc->terminate();
            }
            cv_stop.notify_all();
            // status change will be handled by the managing thread
        } else {
            SRV_WRN("model instance name=%s is not running\n", name.c_str());
        }
    }
}

void server_models::unload_all() {
    std::vector<std::thread> to_join;
    {
        std::lock_guard<std::mutex> lk(mutex);
        for (auto & [name, inst] : mapping) {
            if (inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                SRV_INF("cancelling download for model name=%s\n", name.c_str());
                inst.subproc->stopped.store(true, std::memory_order_relaxed);
            } else if (inst.meta.is_running()) {
                SRV_INF("stopping model instance name=%s\n", name.c_str());
                stopping_models.insert(name);
                cv_stop.notify_all();
                // status change will be handled by the managing thread
            }
            // moving the thread to join list to avoid deadlock
            to_join.push_back(std::move(inst.th));
        }
    }
    for (auto & th : to_join) {
        if (th.joinable()) {
            th.join();
        }
    }
}

void server_models::update_status(const std::string & name, const update_status_args & args) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        auto & meta = it->second.meta;
        meta.status      = args.status;
        meta.exit_code   = args.exit_code;
        if (!args.loaded_info.is_null()) {
            meta.loaded_info = args.loaded_info;
            if (args.status == SERVER_MODEL_STATUS_LOADED) {
                // A ready child has just told us what it actually cost. This is
                // the only moment that measurement exists (issue #36).
                record_measured_footprint(meta);
            }
        }
        if (!args.progress.is_null()) {
            meta.progress = args.progress;
        }
    }
    // broadcast status change to SSE
    {
        json data = {
            {"status", server_model_status_to_string(args.status)},
        };
        if (args.status == SERVER_MODEL_STATUS_UNLOADED) {
            data["exit_code"] = args.exit_code;
        }
        if (!args.loaded_info.is_null()) {
            data["info"] = args.loaded_info;
        }
        if (!args.progress.is_null()) {
            data["progress"] = args.progress;
        }
        // note: notify_sse doesn't acquire the lock, so no deadlock here
        notify_sse("status_change", name, data);
    }
    cv.notify_all();
}

void server_models::update_download_progress(const std::string & name, const common_download_progress & progress, bool done, bool ok) {
    json curr;
    {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            if (done) {
                // mark the instance to be erased on next load_models() call
                it->second.meta.status = SERVER_MODEL_STATUS_DOWNLOADED;
                need_reload = true;
            } else {
                json & info = it->second.meta.loaded_info;
                if (!info.contains("progress")) {
                    info["progress"] = json{};
                }
                info["progress"][progress.url] = {
                    {"done",  progress.downloaded},
                    {"total", progress.total},
                };
                curr = it->second.meta.loaded_info; // copy
            }
        }
    }
    if (done) {
        cv.notify_all(); // notify in case unload() is waiting for download to be cancelled
        notify_sse(ok ? "download_finished" : "download_failed", name, {});
    } else {
        notify_sse("download_progress", name, curr);
    }
}

bool server_models::remove(const std::string & name) {
    // do everything under one lock acquisition; avoid get_meta() /
    // unload() because they can trigger load_models() which erases
    // transient DOWNLOADING / DOWNLOADED entries as a side-effect
    std::unique_lock<std::mutex> lk(mutex);

    auto it = mapping.find(name);
    if (it == mapping.end()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (it->second.meta.source != SERVER_MODEL_SOURCE_CACHE) {
        throw std::runtime_error("model name=" + name + " is not removable (not from cache)");
    }

    if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
        // cancel in-flight download
        SRV_INF("cancelling download for model name=%s\n", name.c_str());
        it->second.subproc->request_exit();
    } else if (it->second.meta.is_running()) {
        // stop running instance
        SRV_INF("stopping model instance name=%s\n", name.c_str());
        stopping_models.insert(name);
        if (it->second.meta.status == SERVER_MODEL_STATUS_LOADING) {
            it->second.subproc->terminate();
        }
        cv_stop.notify_all();
    }

    // wait until the monitoring thread finishes
    wait(lk, name, [](const server_model_meta & meta) {
        return meta.status == SERVER_MODEL_STATUS_UNLOADED
            || meta.status == SERVER_MODEL_STATUS_DOWNLOADED;
    });

    // re-find after wait - load_models() may have erased the entry during the wait
    it = mapping.find(name);
    if (it == mapping.end()) {
        // load_models() already joined the thread and erased the entry;
        // we just need to clean up the cached files on disk
        lk.unlock();
        bool ok = common_download_remove(name);
        SRV_INF("removing model name=%s from cache (%s)\n", name.c_str(), ok ? "succeeded" : "partial");
        notify_sse("model_remove", name, {});
        return true;
    }

    // join before erasing - thread no longer acquires this mutex
    if (it->second.th.joinable()) {
        it->second.th.join();
    }

    // remove from disk (best-effort: cancelled downloads may have no cached files)
    bool ok = common_download_remove(name);
    mapping.erase(name);
    if (!ok) {
        SRV_WRN("removing model name=%s from disk returned false (no cached files?)\n", name.c_str());
    }
    SRV_INF("removing model name=%s from cache (%s)\n", name.c_str(), ok ? "succeeded" : "partial");
    notify_sse("model_remove", name, {});
    return true;
}

void server_models::wait(const std::string & name, std::function<bool(const server_model_meta &)> predicate) {
    std::unique_lock<std::mutex> lk(mutex);
    wait(lk, name, predicate);
}

void server_models::wait(std::unique_lock<std::mutex> & lk, const std::string & name, std::function<bool(const server_model_meta &)> predicate) {
    cv.wait(lk, [this, &name, &predicate]() {
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            return predicate(it->second.meta);

        }
        // model was removed from mapping by another code path (e.g. load_models()).
        // nothing left to wait for - tell the caller to proceed.
        return true;
    });
}

bool server_models::ensure_model_ready(const std::string & name) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (meta->is_ready()) {
        return false; // ready for taking requests
    }
    if (meta->status == SERVER_MODEL_STATUS_SLEEPING) {
        return false; // child is sleeping but still running; new request will wake it up
    }
    if (meta->status == SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model name=%s is not loaded, loading...\n", name.c_str());
        load(name);
    }

    // wait for loading to complete
    SRV_INF("waiting until model name=%s is fully loaded...\n", name.c_str());
    wait(name, [&meta](const server_model_meta & new_meta) {
        if (new_meta.status != SERVER_MODEL_STATUS_LOADING) {
            meta = new_meta; // update meta for final check after wait
            return true;
        }
        return false;
    });

    // check final status
    if (!meta.has_value() || meta->is_failed()) {
        throw std::runtime_error("model name=" + name + " failed to load");
    }

    return true;
}

server_http_res_ptr server_models::proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (!meta->is_running()) {
        throw std::invalid_argument("model name=" + name + " is not running");
    }
    if (update_last_used) {
        std::unique_lock<std::mutex> lk(mutex);
        mapping[name].meta.last_used = ggml_time_ms();
    }
    SRV_INF("proxying request to model %s on port %d\n", name.c_str(), meta->port);
    std::string proxy_path = req.path;
    if (!req.query_string.empty()) {
        proxy_path += '?' + req.query_string;
    }
    auto proxy = std::make_unique<server_http_proxy>(
            method,
            "http",
            CHILD_ADDR,
            meta->port,
            proxy_path,
            req.headers,
            req.body,
            req.files,
            req.should_stop,
            base_params.timeout_read,
            base_params.timeout_write
            );
    return proxy;
}

void server_models::handle_child_state(const std::string & name, const std::string & raw_input) {
    server_state state;
    json payload;

    try {
        json data = json::parse(raw_input.substr(strlen(CMD_CHILD_TO_ROUTER_STATE)));
        state = server_state_from_str(json_value(data, "state", std::string()));
        payload = json_value(data, "payload", json{});
    } catch (const std::exception & e) {
        SRV_ERR("failed to parse child state update for name=%s: %s\n", name.c_str(), e.what());
        return;
    }

    switch (state) {
        case SERVER_STATE_DOWNLOADING:
            {
                std::string result = json_value(payload, "result", std::string());
                std::string url    = json_value(payload, "url",    std::string());
                auto request_exit = [&]() {
                    std::lock_guard<std::mutex> lk(mutex);
                    auto it = mapping.find(name);
                    if (it != mapping.end()) {
                        return it->second.subproc->request_exit();
                    }
                };
                if (result == "download_finished") {
                    update_download_progress(name, {}, true, true);
                    request_exit();
                } else if (result == "download_failed") {
                    update_download_progress(name, {}, true, false);
                    request_exit();
                } else if (!url.empty()) {
                    common_download_progress p;
                    p.url        = url;
                    p.downloaded = json_value(payload, "downloaded", (size_t)0);
                    p.total      = json_value(payload, "total", (size_t)0);
                    update_download_progress(name, p, false);
                }
            } break;
        case SERVER_STATE_LOADING:
            {
                update_status(name, {
                    SERVER_MODEL_STATUS_LOADING,
                    0,
                    nullptr, // no loaded_info yet
                    payload,
                });
            } break;
        case SERVER_STATE_READY:
            {
                update_status(name, {
                    SERVER_MODEL_STATUS_LOADED,
                    0,
                    // note: payload can be empty if this is a wakeup from sleep
                    payload.size() > 0 ? payload : nullptr,
                    {}, // reset progress info
                });
            } break;
        case SERVER_STATE_SLEEPING:
            {
                update_status(name, { SERVER_MODEL_STATUS_SLEEPING });
            } break;
        default:
            // should never happen, but just in case
            GGML_ASSERT(false && "unexpected state from child server");
    }
}

//
// server_child
//

bool server_child::is_child() {
    const char * router_port = std::getenv("LLAMA_SERVER_ROUTER_PORT");
    return router_port != nullptr;
}

server_child_mode server_child::get_mode() {
    const char * mode = std::getenv("LLAMA_SERVER_CHILD_MODE");
    std::string mode_str(mode ? mode : "");
    if (mode_str == "download") {
        return SERVER_CHILD_MODE_DOWNLOAD;
    } else {
        return SERVER_CHILD_MODE_NORMAL;
    }
}

struct server_download_state : public common_download_callback {
    server_child * self;
    std::function<bool()> should_stop;
    std::atomic<int64_t> last_progress_time{0}; // multiple files downloading in different threads
    bool is_ok = false;

    server_download_state(server_child * s) : self(s) {}

    bool run(common_params & params) {
        try {
            common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
            common_models_handler_apply(handler, params, this);
            is_ok = true;
        } catch (const std::exception & e) {
            auto model_name = params.model.get_name();
            SRV_ERR("download failed for model name=%s: %s\n", model_name.c_str(), e.what());
            is_ok = false;
        }
        return is_ok;
    }
    void on_progress(const common_download_progress & p) {
        json data = {
            {"url", p.url},
            {"downloaded", p.downloaded},
            {"total", p.total},
        };
        self->notify_to_router(server_state_to_str(SERVER_STATE_DOWNLOADING), data);
    }
    void on_start(const common_download_progress & p) override {
        on_progress(p);
    }
    void on_update(const common_download_progress & p) override {
        int64_t now = ggml_time_ms();
        // throttle progress updates to avoid flooding logs
        if (now - last_progress_time.load(std::memory_order_relaxed) >= 100) {
            on_progress(p);
            last_progress_time.store(now, std::memory_order_relaxed);
        }
    }
    void on_done(const common_download_progress & p, bool) override {
        on_progress(p);
    }
    bool is_cancelled() const override {
        return should_stop ? should_stop() : false;
    }
};

int server_child::run_download(common_params & params) {
    auto cancelled = std::make_shared<std::atomic<bool>>(false);

    // monitor stdin for cancellation command from the router
    std::thread signal_thread = setup([cancelled](int) {
        cancelled->store(true, std::memory_order_relaxed);
    });

    server_download_state dl(this);
    dl.should_stop = [cancelled]() {
        return cancelled->load(std::memory_order_relaxed);
    };

    bool ok = dl.run(params);

    notify_to_router(server_state_to_str(SERVER_STATE_DOWNLOADING), {
        {"result", ok ? "download_finished" : "download_failed"},
    });

    // router should send CMD_ROUTER_TO_CHILD_EXIT after receiving the result
    if (signal_thread.joinable()) {
        signal_thread.join();
    }

    SRV_INF("download completed %s\n", ok ? "successfully" : "with errors");
    return 0;
}

std::thread server_child::setup(const std::function<void(int)> & shutdown_handler) {
    // setup thread for monitoring stdin
    return std::thread([shutdown_handler]() {
        // wait for EOF on stdin
        SRV_INF("%s", "child server monitoring thread started, waiting for EOF on stdin...\n");
        bool eof = false;
        while (true) {
            std::string line;
            if (!std::getline(std::cin, line)) {
                // EOF detected, that means the router server is unexpectedly exit or killed
                eof = true;
                break;
            }
            if (line.find(CMD_ROUTER_TO_CHILD_EXIT) != std::string::npos) {
                SRV_INF("%s", "exit command received, exiting...\n");
                shutdown_handler(0);
                break;
            }
        }
        if (eof) {
            SRV_INF("%s", "EOF on stdin detected, forcing shutdown...\n");
            exit(1);
        }
    });
}

void server_child::notify_to_router(const std::string & state, const json & payload) {
    json data = {
        {"state", state},
        {"payload", payload},
    };
    std::lock_guard<std::mutex> lk(mtx_stdout);
    common_log_pause(common_log_main());
    fflush(stdout);
    fprintf(stdout, "%s%s\n", CMD_CHILD_TO_ROUTER_STATE, safe_json_to_str(data).c_str());
    fflush(stdout);
    common_log_resume(common_log_main());
}


//
// server_models_routes
//

// RAII wrapper similar to server_response_reader, but doesn't use server_queue
static std::atomic<int> sse_client_id_counter = 0;
struct server_models_sse_client {
    server_response & queue_results;
    int client_id;
    server_models_sse_client(server_response & q)
            : queue_results(q), client_id(sse_client_id_counter.fetch_add(1, std::memory_order_relaxed)) {
        SRV_DBG("new SSE client connected, assigned client_id=%d\n", client_id);
        queue_results.add_waiting_task_id(client_id);
    }
    ~server_models_sse_client() {
        SRV_DBG("SSE client disconnected, removing client_id=%d\n", client_id);
        queue_results.remove_waiting_task_id(client_id);
    }

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop) {
        while (true) {
            static const int http_polling_seconds = 1; // check should_stop every 1 second
            server_task_result_ptr result = queue_results.recv_with_timeout({client_id}, http_polling_seconds);
            if (result == nullptr) {
                // timeout, check stop condition
                if (should_stop()) {
                    return nullptr;
                }
                // continue waiting otherwise
            } else {
                SRV_DBG("recv result for client_id=%d: %s\n", client_id, safe_json_to_str(result->to_json()).c_str());
                return result;
            }
        }
        // should not reach here
    }
};

static void res_ok(std::unique_ptr<server_http_res> & res, const json & response_data) {
    res->status = 200;
    res->data = safe_json_to_str(response_data);
}

static void res_err(std::unique_ptr<server_http_res> & res, const json & error_data) {
    res->status = json_value(error_data, "code", 500);
    res->data = safe_json_to_str({{ "error", error_data }});
}

static bool router_validate_model(std::string & name, server_models & models, bool models_autoload, std::unique_ptr<server_http_res> & res) {
    if (name.empty()) {
        res_err(res, format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    auto meta = models.get_meta(name);
    if (!meta.has_value()) {
        res_err(res, format_error_response(string_format("model '%s' not found", name.c_str()), ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    // resolve alias to canonical model name
    name = meta->name;
    if (models_autoload) {
        models.ensure_model_ready(name);
    } else {
        if (!meta->is_running()) {
            res_err(res, format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
            return false;
        }
    }
    return true;
}

static bool is_autoload(const common_params & params, const server_http_req & req) {
    std::string autoload = req.get_param("autoload");
    if (autoload.empty()) {
        return params.models_autoload;
    } else {
        return autoload == "true" || autoload == "1";
    }
}

// percent encode one query or path component, covers reserved chars without pulling in
// httplib::detail. used by the stream routes to forward conversation_id to children safely
static std::string encode_qs(const std::string & in) {
    std::string out;
    out.reserve(in.size() * 3);
    for (unsigned char c : in) {
        bool safe = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
                 || c == '-' || c == '_' || c == '.' || c == '~';
        if (safe) {
            out.push_back(char(c));
        } else {
            char buf[4];
            std::snprintf(buf, sizeof(buf), "%%%02X", c);
            out.append(buf, 3);
        }
    }
    return out;
}

// resolve the child that owns a conversation's stream session via the conv_id -> model map
// populated when the POST was routed. single map lookup then a meta lookup, no polling, no
// parsing of the conv id. returns nullopt when nothing maps, the caller answers not found and
// the client recovers
static std::optional<server_model_meta> resolve_child_for_conv(
        server_models & models, const std::string & conversation_id) {
    if (conversation_id.empty()) {
        return std::nullopt;
    }
    auto tracked = models.conv_models.lookup(conversation_id);
    if (!tracked.has_value()) {
        return std::nullopt;
    }
    auto meta = models.get_meta(*tracked);
    if (meta.has_value() && meta->is_ready()) {
        return meta;
    }
    return std::nullopt;
}

void server_models_routes::init_routes() {
    this->get_router_props = [this](const server_http_req & req) {
        std::string name = req.get_param("model");
        if (name.empty()) {
            // main instance
            auto res = std::make_unique<server_http_res>();
            res_ok(res, {
                // TODO: add support for this on web UI
                {"role",                 "router"},
                {"max_instances",        params.models_max},
                {"models_autoload",      params.models_autoload},
                {"models_vram_aware",         params.models_vram_aware},
                {"models_vram_margin_mb",     params.models_vram_margin_mb},
                {"models_vram_overhead_pct",  params.models_vram_overhead_pct},
                // this is a dummy response to make sure the UI doesn't break
                {"model_alias", "llama-server"},
                {"model_path",  "none"},
                {"default_generation_settings", {
                    {"params", json{}},
                    {"n_ctx",  0},
                }},
                // New key
                {"ui_settings",          ui_settings},
                {"build_info",           std::string(llama_build_info())},
                {"cors_proxy_enabled",   params.ui_mcp_proxy},
            });
            return res;
        }
        return proxy_get(req);
    };

    this->proxy_get = [this](const server_http_req & req) {
        std::string method = "GET";
        std::string name = req.get_param("model");
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        return models.proxy_request(req, method, name, false);
    };

    this->proxy_post = [this](const server_http_req & req) {
        std::string method = "POST";
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        if (name.empty()) {
            // self.llamolotl#44. Taking the target model ONLY from the body
            // makes every proxied POST route whose child requires a non-object
            // body unreachable, because there is nowhere to put "model".
            //
            // POST /lora-adapters is exactly that: post_lora_adapters
            // (server-context.cpp) rejects any body that is not a JSON array,
            // while proxy_request forwards the body to the child VERBATIM. So
            // an array 400s here for a missing model, and an object carrying
            // one 400s at the child for not being an array. No body satisfies
            // both, and runtime LoRA scale changes are unreachable in router
            // mode as a result.
            //
            // Falling back to the ?model= query parameter -- the same place
            // proxy_get already reads it from -- resolves that without
            // touching the body, so the child still receives precisely the
            // payload its handler expects. Body-carried "model" keeps
            // priority, so no existing caller changes behaviour.
            name = req.get_param("model");
        }
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        // remember which child serves this conversation so the stream routes can route straight
        // to it without polling, keyed on the exact conv id from the header
        std::string conv_id = server_stream_conv_id_from_headers(req.headers);
        if (!conv_id.empty()) {
            models.conv_models.remember(conv_id, name);
        }
        return models.proxy_request(req, method, name, true); // update last usage for POST request only
    };

    this->post_router_models_load = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto meta = models.get_meta(name);
        if (!meta.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        if (meta->is_running()) {
            res_err(res, format_error_response("model is already running", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.load(meta->name);
        res_ok(res, {{"success", true}});
        return res;
    };

    this->get_router_models = [this](const server_http_req & req) {
        bool reload = !req.get_param("reload", "").empty();
        if (reload) {
            models.load_models();
        }
        auto res = std::make_unique<server_http_res>();
        json models_json = json::array();
        auto all_models = models.get_all_meta();
        std::time_t t = std::time(0);
        for (const auto & meta : all_models) {
            json status {
                {"value",  server_model_status_to_string(meta.status)},
                {"args",   meta.args},
            };
            if (!meta.preset.name.empty()) {
                common_preset preset_copy = meta.preset;
                unset_reserved_args(preset_copy, false);
                preset_copy.unset_option("LLAMA_ARG_HOST");
                preset_copy.unset_option("LLAMA_ARG_PORT");
                preset_copy.unset_option("LLAMA_ARG_ALIAS");
                preset_copy.unset_option("LLAMA_ARG_TAGS");
                status["preset"] = preset_copy.to_ini();
            }
            if (meta.is_failed()) {
                status["exit_code"] = meta.exit_code;
                status["failed"]    = true;
            }

            // pi coding agent multimodal compatibility
            json input_modalities = json::array({"text"});
            if (meta.multimodal.inp_vision) {
                input_modalities.push_back("image");
            }
            if (meta.multimodal.inp_audio) {
                input_modalities.push_back("audio");
            }
            json architecture {
                {"input_modalities",  input_modalities},
                {"output_modalities", json::array({"text"})},
            };

            json model_info = json {
                {"id",            meta.name},
                {"aliases",       meta.aliases},
                {"tags",          meta.tags},
                {"object",        "model"},    // for OAI-compat
                {"owned_by",      "llamacpp"}, // for OAI-compat
                {"created",       t},          // for OAI-compat
                {"status",        status},
                {"architecture",  architecture},
                {"source",        server_model_source_to_string(meta.source)},
                {"can_remove",    meta.source == SERVER_MODEL_SOURCE_CACHE},
                // {"need_download", meta.need_download},
                // TODO: add other fields, may require reading GGUF metadata
            };

            // How much of the card this configuration takes, and how we know
            // (self.ai#107). OMITTED ENTIRELY when unknown -- a `null` or a 0
            // here would read as "this model is free", and a consumer sizing a
            // VRAM lease off that would ask for nothing and be granted it.
            // Absence is the honest encoding of absence.
            //
            // This exists so self.ai can request a broker lease of the right
            // size BEFORE asking us to load. Without it core has no figure at
            // all, so its only options are to guess or to skip the broker --
            // and skipping is what left lower-priority holders immovable while
            // llamolotl, the highest-priority consumer, OOM'd around them.
            const footprint_info fp = models.footprint_report(meta);
            if (fp.bytes > 0 && fp.source != nullptr) {
                model_info["vram_footprint"] = json {
                    {"bytes",  fp.bytes},
                    {"source", fp.source},
                };
            }

            // merge with loaded_info from the child process if available
            if (meta.is_running()) {
                for (auto it = meta.loaded_info.begin(); it != meta.loaded_info.end(); ++it) {
                    if (!model_info.contains(it.key())) {
                        model_info[it.key()] = it.value();
                    }
                }
            }
            models_json.push_back(model_info);
        }
        res_ok(res, {
            {"data", models_json},
            {"object", "list"},
        });
        return res;
    };

    this->post_router_models_unload = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto model = models.get_meta(name);
        if (!model.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!model->is_running() && model->status != SERVER_MODEL_STATUS_DOWNLOADING) {
            res_err(res, format_error_response("model is not running", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.unload(model->name);
        res_ok(res, {{"success", true}});
        return res;
    };

    this->get_router_models_sse = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->content_type = "text/event-stream";
        auto sse_client = std::make_shared<server_models_sse_client>(models.sse);
        res->next = [this, sse_client, &req](std::string & output) -> bool {
            auto result = sse_client->next([&]() {
                return stopping.load(std::memory_order_relaxed) || req.should_stop();
            });
            if (result == nullptr) {
                return false; // client disconnected or should_stop
            }
            output = "data: " + safe_json_to_str(result->to_json()) + "\n\n";
            return true; // listen for the next event
        };
        return res;
    };

    this->post_router_models = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();

        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        if (name.empty()) {
            throw std::invalid_argument("model must be a non-empty string");
        }

        common_params p;
        p.model.hf_repo  = name;
        p.hf_token       = params.hf_token;

        // validate by fetching metadata
        bool ok = false;
        try {
            common_models_handler_init(p, LLAMA_EXAMPLE_SERVER);
            ok = true;
        } catch (...) {
            SRV_ERR("unknown error while validating model '%s'\n", name.c_str());
            // other exceptions will be handled by the outer ex_wrapper()
            throw;
        }

        if (!ok) {
            throw std::invalid_argument("model validation failed, unable to download");
        }

        // reject if model already exists
        if (models.has_model(name)) {
            throw std::invalid_argument("model '" + name + "' already exists");
        }

        // then, proceed with the actual download
        SRV_INF("starting download for model '%s'\n", name.c_str());
        {
            server_models::load_options load_opts;
            load_opts.mode = SERVER_CHILD_MODE_DOWNLOAD;
            load_opts.custom_meta = server_model_meta{};
            load_opts.custom_meta->source = SERVER_MODEL_SOURCE_CACHE;
            load_opts.custom_meta->name   = name;
            models.load(name, load_opts);
        }

        res_ok(res, {{"success", true}});
        return res;
    };

    this->del_router_models = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();

        std::string name = req.get_param("model");
        if (name.empty()) {
            throw std::invalid_argument("model must be a non-empty string");
        }

        models.remove(name); // throws on error

        res_ok(res, {{"success", true}});
        return res;
    };

    this->router_stream_get = [this](const server_http_req & req) {
        // GET /v1/stream/<conv_id>?from=N. resolve the owning child from the conv_id -> model
        // map, 404 when nothing maps
        auto res = std::make_unique<server_http_res>();
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            res_err(res, format_error_response("Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::optional<server_model_meta> owner = resolve_child_for_conv(models, conv_id);
        if (!owner.has_value()) {
            res_err(res, format_error_response("Stream not found or expired", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        std::string from = req.get_param("from");
        std::string child_path = "/v1/stream/" + encode_qs(conv_id);
        if (!from.empty()) {
            child_path += "?from=" + from;
        }
        SRV_TRC("proxying stream resume to model %s on port %d, path=%s\n",
                owner->name.c_str(), owner->port, child_path.c_str());
        auto proxy = std::make_unique<server_http_proxy>(
                "GET",
                "http",
                CHILD_ADDR,
                owner->port,
                child_path,
                req.headers,
                req.body,
                req.files,
                req.should_stop,
                params.timeout_read,
                params.timeout_write);
        return std::unique_ptr<server_http_res>(std::move(proxy));
    };

    this->router_streams_lookup = [this](const server_http_req & req) {
        // POST /v1/streams/lookup. resolve each requested conv id to its owning child via the
        // map, group the ids per child, and query only the children that actually own some of
        // them instead of fanning out to every ready child. a child only answers for the ids
        // it owns, never lists anything else
        auto res = std::make_unique<server_http_res>();
        std::vector<std::string> requested;
        try {
            json body = json::parse(req.body);
            if (body.contains("conversation_ids") && body["conversation_ids"].is_array()) {
                for (const auto & v : body["conversation_ids"]) {
                    if (v.is_string() && !v.get<std::string>().empty()) {
                        requested.push_back(v.get<std::string>());
                    }
                }
            }
        } catch (const std::exception &) {
            res_ok(res, json::array());
            return res;
        }

        // group requested ids by the child port that owns them, drop ids that map to nothing
        std::unordered_map<int, json> per_child;
        for (const auto & cid : requested) {
            auto owner = resolve_child_for_conv(models, cid);
            if (!owner.has_value()) {
                continue;
            }
            per_child[owner->port].push_back(cid);
        }

        json aggregated = json::array();
        for (auto & [port, ids] : per_child) {
            json child_body = {{"conversation_ids", ids}};
            httplib::Client cli(CHILD_ADDR, port);
            cli.set_connection_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_read_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_write_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            auto resp = cli.Post("/v1/streams/lookup", child_body.dump(), "application/json");
            if (!resp || resp->status != 200) {
                continue;
            }
            try {
                json child_arr = json::parse(resp->body);
                if (!child_arr.is_array()) {
                    continue;
                }
                for (auto & entry : child_arr) {
                    if (entry.is_object()) {
                        aggregated.push_back(entry);
                    }
                }
            } catch (const std::exception &) {
                continue;
            }
        }
        res_ok(res, aggregated);
        return res;
    };

    this->router_stream_delete = [this](const server_http_req & req) {
        // DELETE /v1/stream/<conv_id>. resolve the owning child via the map and forward only to
        // it, evict_and_cancel is idempotent on the child
        auto res = std::make_unique<server_http_res>();
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            res_err(res, format_error_response("Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::string child_path = "/v1/stream/" + encode_qs(conv_id);
        auto owner = resolve_child_for_conv(models, conv_id);
        if (owner.has_value()) {
            httplib::Client cli(CHILD_ADDR, owner->port);
            cli.set_connection_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_read_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_write_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            auto resp = cli.Delete(child_path.c_str());
            (void) resp; // the child logs its own miss when the session is unknown there
        } else {
            SRV_WRN("router stop for unknown conv_id=%s, no owning child in the conv map\n",
                    conv_id.c_str());
        }
        // drop the tracking entry, the session is being torn down
        models.conv_models.forget(conv_id);
        res->status = 204;
        res->content_type = "application/json";
        return res;
    };
}



//
// server_http_proxy
//

// simple implementation of a pipe
// used for streaming data between threads
template<typename T>
struct pipe_t {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<T> queue;
    std::atomic<bool> writer_closed{false};
    std::atomic<bool> reader_closed{false};
    void close_write() {
        writer_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }
    void close_read() {
        reader_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }
    bool read(T & output, const std::function<bool()> & should_stop) {
        std::unique_lock<std::mutex> lk(mutex);
        constexpr auto poll_interval = std::chrono::milliseconds(500);
        while (true) {
            if (!queue.empty()) {
                output = std::move(queue.front());
                queue.pop();
                return true;
            }
            if (writer_closed.load()) {
                return false; // clean EOF
            }
            if (should_stop()) {
                close_read(); // signal broken pipe to writer
                return false; // cancelled / reader no longer alive
            }
            cv.wait_for(lk, poll_interval);
        }
    }
    bool write(T && data) {
        std::lock_guard<std::mutex> lk(mutex);
        if (reader_closed.load()) {
            return false; // broken pipe
        }
        queue.push(std::move(data));
        cv.notify_one();
        return true;
    }
};

static std::string to_lower_copy(const std::string & value) {
    std::string lowered(value.size(), '\0');
    std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
    return lowered;
}

static bool should_strip_proxy_header(const std::string & header_name) {
    // Headers that get duplicated when router forwards child responses
    if (header_name == "server" ||
        header_name == "transfer-encoding" ||
        header_name == "content-length" || // quick fix for https://github.com/ggml-org/llama.cpp/issues/17710
        header_name == "keep-alive") {
        return true;
    }

    // Router injects CORS, child also sends them: duplicate
    if (header_name.rfind("access-control-", 0) == 0) {
        return true;
    }

    return false;
}

static std::string generate_multipart_boundary() {
    thread_local std::mt19937 gen(std::random_device{}());
    static const char chars[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<> dis(0, sizeof(chars) - 2);
    std::string boundary = "----llama-cpp-proxy-";
    for (int i = 0; i < 16; i++) {
        boundary += chars[dis(gen)];
    }
    return boundary;
}

static std::string build_multipart_body(
        const json & form_fields,
        const std::map<std::string, uploaded_file> & files,
        const std::string & boundary) {
    static auto sanitize_field = [](const std::string & text) {
        std::string result;
        result.reserve(text.size());
        for (char c : text) {
            if (c != '\n' && c != '\r' && c != '"') {
                result += c;
            }
        }
        return result;
    };

    std::ostringstream body;

    for (const auto & [key, value] : form_fields.items()) {
        if (value.is_array()) {
            for (const auto & item : value) {
                body << "--" << boundary << "\r\n";
                body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"\r\n";
                body << "\r\n";
                if (!item.is_string()) {
                    throw std::invalid_argument("expected string");
                }
                body << item.get<std::string>() << "\r\n";
            }
        } else {
            body << "--" << boundary << "\r\n";
            body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"\r\n";
            body << "\r\n";
            if (!value.is_string()) {
                throw std::invalid_argument("expected string");
            }
            body << value.get<std::string>() << "\r\n";
        }
    }

    for (const auto & [key, file] : files) {
        body << "--" << boundary << "\r\n";
        body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"";
        if (!file.filename.empty()) {
            body << "; filename=\"" << sanitize_field(file.filename) << "\"";
        }
        body << "\r\n";
        if (!file.content_type.empty()) {
            body << "Content-Type: " << sanitize_field(file.content_type) << "\r\n";
        } else {
            body << "Content-Type: application/octet-stream\r\n";
        }
        body << "\r\n";
        body.write(reinterpret_cast<const char*>(file.data.data()), file.data.size());
        body << "\r\n";
    }

    body << "--" << boundary << "--\r\n";
    return body.str();
}

server_http_proxy::server_http_proxy(
        const std::string & method,
        const std::string & scheme,
        const std::string & host,
        int port,
        const std::string & path,
        const std::map<std::string, std::string> & headers,
        const std::string & body,
        const std::map<std::string, uploaded_file> & files,
        const std::function<bool()> should_stop,
        int32_t timeout_read,
        int32_t timeout_write
        ) {
    // shared between reader and writer threads
    auto cli  = std::make_shared<httplib::ClientImpl>(host, port);
    auto pipe = std::make_shared<pipe_t<msg_t>>();

    if (scheme == "https") {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        cli.reset(new httplib::SSLClient(host, port));
#else
        throw std::runtime_error("HTTPS requested but CPPHTTPLIB_OPENSSL_SUPPORT is not defined");
#endif
    }

    // setup Client
    cli->set_follow_location(true);
    cli->set_connection_timeout(timeout_read, 0); // use --timeout value instead of hardcoded 5 s
    cli->set_write_timeout(timeout_read, 0); // reversed for cli (client) vs srv (server)
    cli->set_read_timeout(timeout_write, 0);
    this->status = 500; // to be overwritten upon response
    this->cleanup = [pipe]() {
        pipe->close_read();
        pipe->close_write();
    };

    // wire up the receive end of the pipe
    this->next = [pipe, should_stop](std::string & out) -> bool {
        msg_t msg;
        bool has_next = pipe->read(msg, should_stop);
        if (!msg.data.empty()) {
            out = std::move(msg.data);
        }
        return has_next; // false if EOF or pipe broken
    };

    // wire up the HTTP client
    // note: do NOT capture `this` pointer, as it may be destroyed before the thread ends
    httplib::ResponseHandler response_handler = [pipe, cli](const httplib::Response & response) {
        msg_t msg;
        msg.status = response.status;
        for (const auto & [key, value] : response.headers) {
            const auto lowered = to_lower_copy(key);
            if (should_strip_proxy_header(lowered)) {
                continue;
            }
            if (lowered == "content-type") {
                msg.content_type = value;
                continue;
            }
            msg.headers[key] = value;
        }
        return pipe->write(std::move(msg)); // send headers first
    };
    httplib::ContentReceiverWithProgress content_receiver = [pipe](const char * data, size_t data_length, size_t, size_t) {
        // send data chunks
        // returns false if pipe is closed / broken (signal to stop receiving)
        return pipe->write({{}, 0, std::string(data, data_length), ""});
    };

    // when files are present, the body was converted from multipart form data to JSON
    // we need to reconstruct the multipart body for the downstream server
    std::string effective_body = body;
    std::string override_content_type;
    bool has_files = !files.empty();

    if (has_files) {
        json form_fields = json::parse(body, nullptr, false);
        if (!form_fields.is_discarded()) {
            auto boundary = generate_multipart_boundary();
            effective_body = build_multipart_body(form_fields, files, boundary);
            override_content_type = "multipart/form-data; boundary=" + boundary;
        } else {
            throw std::runtime_error("failed to parse multipart form fields JSON");
        }
    }

    // prepare the request to destination server
    httplib::Request req;
    {
        req.method = method;
        req.path = path;
        for (const auto & [key, value] : headers) {
            const auto lowered = to_lower_copy(key);
            if (lowered == "accept-encoding") {
                // disable Accept-Encoding to avoid compressed responses
                continue;
            }
            if (lowered == "transfer-encoding") {
                // the body is already decoded
                continue;
            }
            if (lowered == "content-length") {
                // let httplib calculate Content-Length from the actual body
                continue;
            }
            if (lowered == "content-type") {
                if (has_files) {
                    // we set our own Content-Type with the new boundary
                    continue;
                }
                // when no files but the original request was multipart,
                // the body is now JSON, so correct the Content-Type
                if (value.find("multipart/form-data") != std::string::npos) {
                    override_content_type = "application/json; charset=utf-8";
                    continue;
                }
            }
            if (lowered == "host") {
                bool is_default_port = (scheme == "https" && port == 443) || (scheme == "http" && port == 80);
                const std::string url_host = common_http_format_host(host);
                req.set_header(key, is_default_port ? url_host : url_host + ":" + std::to_string(port));
            } else {
                req.set_header(key, value);
            }
        }
        req.body = effective_body;
        if (!override_content_type.empty()) {
            req.set_header("Content-Type", override_content_type);
        }
        req.response_handler = response_handler;
        req.content_receiver = content_receiver;
    }

    // start the proxy thread
    SRV_DBG("start proxy thread %s %s\n", req.method.c_str(), req.path.c_str());
    this->thread = std::thread([cli, pipe, req]() {
        auto result = cli->send(std::move(req));
        if (result.error() != httplib::Error::Success) {
            auto err_str = httplib::to_string(result.error());
            SRV_ERR("http client error: %s\n", err_str.c_str());
            pipe->write({{}, 500, "", ""}); // header
            pipe->write({{}, 0, "proxy error: " + err_str, ""}); // body
        }
        pipe->close_write(); // signal EOF to reader
        SRV_DBG("%s", "client request thread ended\n");
    });
    this->thread.detach();

    // wait for the first chunk (headers)
    {
        msg_t header;
        if (pipe->read(header, should_stop)) {
            SRV_DBG("%s", "received response headers\n");
            this->status  = header.status;
            this->headers = std::move(header.headers);
            if (!header.content_type.empty()) {
                this->content_type = std::move(header.content_type);
            }
        } else {
            SRV_DBG("%s", "no response headers received (request cancelled?)\n");
        }
    }
}
