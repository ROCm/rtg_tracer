class Flag {
    const char *name_;
    const char *value_;
    const char *default_;
    const char *desc_;
    bool value_bool_;
    bool is_bool_;
    static std::vector<Flag*> flags;

public:

    Flag(const char *name, const char* _default, const char *desc, bool is_bool)
    : name_(name), value_(NULL), default_(_default), desc_(desc), value_bool_(false), is_bool_(is_bool)
    {
        flags.push_back(this);
    }

    void init() {
        value_ = getenv(name_);
        if (NULL == value_) value_ = default_;
        if (is_bool_) {
            value_bool_ = value_[0] == 'T' || value_[0] == 't' || value_[0] == 'Y' || value_[0] == 'y';
            value_bool_ |= (isdigit(value_[0]) && value_[0] != '0');
        }
    }

    void print() {
        fprintf(stderr, "RTG Tracer: %-25s = %-15s %s\n", name_, value_, desc_);
    }

    static void init_all();

    operator std::string() { return value_; }
    bool operator==(const std::string& that) { return std::string(value_) == that; }
    operator bool() { return value_bool_; }
    void operator=(bool value) { value_bool_ = value; }
    bool empty() { return value_ == NULL ? true : strlen(value_) == 0; }

};

std::vector<Flag*> Flag::flags;

#define FLAG_BOOL(name, default_, desc) Flag name(#name, #default_, desc, true)
#define FLAG_CHAR(name, default_, desc) Flag name(#name, default_, desc, false)

#ifdef RPD_TRACER
FLAG_BOOL(RTG_RPD, 1, "write output in rpd format");
#endif

FLAG_BOOL(RTG_VERBOSE, 0, "Verbose logging from RTG");
FLAG_CHAR(RTG_FILENAME, "rtg_trace.txt", "Output filename, default rtg_trace.txt -- pid always appended");
FLAG_BOOL(RTG_HIP_API_ARGS, false, "Capture HIP API name and function arguments, otherwise just the name");
FLAG_CHAR(RTG_HIP_API_FILTER, "all", "Trace specific HIP calls. Special case 'all', otherwise simple string matching. Separate tokens with ','");
FLAG_CHAR(RTG_HIP_API_FILTER_OUT, "", "Do not trace specific HIP calls. Simple string matching. Separate tokens with ','");
FLAG_BOOL(HCC_PROFILE, 0, "Legacy HCC profiling, for use with rpt tool");
FLAG_BOOL(RTG_PROFILE, true, "Enable profiling of device kernels and barriers");
FLAG_BOOL(RTG_PROFILE_COPY, true, "Enable profiling of device async copy operations (noticable overhead)");
FLAG_CHAR(RTG_HSA_API_FILTER, "", "Trace specific HSA calls. Special case 'all', 'core', and 'ext', otherwise simple string matching. Separate tokens with ','");
FLAG_CHAR(RTG_HSA_API_FILTER_OUT, "", "Do not trace specific HSA calls. Simple string matching. Separate tokens with ','");
FLAG_BOOL(RTG_HSA_HOST_DISPATCH, false, "Trace when kernel dispatch is enqueued on the host");
FLAG_BOOL(RTG_DEMANGLE, true, "Demangle kernel names");
FLAG_BOOL(RTG_LEGACY_PRINTF, false, "use the old printf logger (will have periodic stalls due to logging)");
// Not RTG exactly, but still parsed by RTG.
FLAG_CHAR(HIP_VISIBLE_DEVICES, "", "Devices ordinals visible to HIP");
FLAG_CHAR(CUDA_VISIBLE_DEVICES, "", "Devices ordinals visible to HIP, but using the CUDA version of the env var");

void Flag::init_all() {
    for (auto flag : flags) {
        flag->init();
    }

    // Flags that influence other flags
    if (HCC_PROFILE) {
        RTG_HIP_API_FILTER = "";
        RTG_HIP_API_ARGS = false;
        RTG_PROFILE = true;
        RTG_PROFILE_COPY = true;
        RTG_HSA_API_FILTER = "";
        RTG_HSA_HOST_DISPATCH = false;
    }
    else {
        if (RTG_HSA_API_FILTER.empty() && !RTG_HSA_API_FILTER_OUT.empty()) {
            RTG_HSA_API_FILTER = "all";
        }

        if (RTG_HIP_API_FILTER.empty() && !RTG_HIP_API_FILTER_OUT.empty()) {
            RTG_HIP_API_FILTER = "all";
        }
    }

    if (RTG_VERBOSE) {
        fprintf(stderr, "RTG Tracer: Settings ------------------------------------\n");
        for (auto flag : flags) {
            flag->print();
        }
        fprintf(stderr, "RTG Tracer: ---------------------------------------------\n");
    }
}

