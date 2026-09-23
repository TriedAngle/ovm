pub use tracing::{Level, debug_span, info_span, instrument, trace_span, warn_span};

pub fn init() {
    let filter = match std::env::var("RUST_LOG") {
        Ok(v) if !v.is_empty() => v,
        _ => match std::env::var("TRACE") {
            Ok(v) if v == "1" => "info".to_string(),
            _ => return,
        },
    };
    use tracing_subscriber::fmt;
    let build = || {
        fmt()
            .with_span_events(fmt::format::FmtSpan::ENTER | fmt::format::FmtSpan::CLOSE)
            .with_target(false)
            .with_env_filter(tracing_subscriber::EnvFilter::new(&filter))
            .finish()
    };
    let _ = tracing::subscriber::set_global_default(build());
}
