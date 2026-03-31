use std::ffi::OsString;

/// Python-style single-dash aliases accepted for `vllm-rs serve` before explicit passthrough.
///
/// These are rewritten to their canonical Rust-owned long flags before `bpaf` parses argv.
const SERVE_LEGACY_ALIASES: &[(&str, &str)] = &[
    ("-dp", "--data-parallel-size"),
    ("-dpa", "--data-parallel-address"),
    ("-dpp", "--data-parallel-rpc-port"),
];

/// Split one argv sequence into the program name and the remaining arguments.
///
/// `bpaf::Args` expects argv without the executable name, but we still keep the original name so
/// generated help and error messages use the invoked binary name.
pub(super) fn split_name_from_args<I, T>(itr: I) -> (String, Vec<OsString>)
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    let args: Vec<OsString> = itr.into_iter().map(Into::into).collect();
    let name = args
        .first()
        .map(|arg| arg.to_string_lossy().into_owned())
        .unwrap_or_else(|| "vllm-rs".to_string());
    (name, args.into_iter().skip(1).collect())
}

/// Rewrite legacy single-dash `serve` aliases into canonical long flags.
///
/// The rewrite only applies:
/// - after the `serve` subcommand is seen
/// - before an explicit `--` passthrough boundary
///
/// Once users opt into explicit passthrough with `--`, the remaining argv is left untouched and
/// forwarded to Python exactly as written.
pub(super) fn rewrite_serve_legacy_aliases(args: Vec<OsString>) -> Vec<OsString> {
    let mut rewritten = Vec::with_capacity(args.len());
    let mut seen_serve = false;
    let mut passthrough = false;

    for arg in args {
        if !seen_serve {
            if arg == "serve" {
                seen_serve = true;
            }
            rewritten.push(arg);
            continue;
        }

        if passthrough {
            rewritten.push(arg);
            continue;
        }

        if arg == "--" {
            passthrough = true;
            rewritten.push(arg);
            continue;
        }

        rewritten.push(rewrite_legacy_alias(arg));
    }

    rewritten
}

/// Keep passthrough capture from consuming `bpaf`'s built-in help flags.
///
/// This lets `serve --help` and `serve -h` continue to render CLI help instead of forwarding those
/// tokens into Python passthrough.
pub(super) fn not_help(arg: String) -> Option<String> {
    (!matches!(arg.as_str(), "--help" | "-h")).then_some(arg)
}

/// Rewrite one legacy single-dash alias token into its canonical long form.
///
/// Both `-dpa value` and `-dpa=value` spellings are supported.
fn rewrite_legacy_alias(arg: OsString) -> OsString {
    let arg_str = arg.to_string_lossy();
    for (alias, canonical) in SERVE_LEGACY_ALIASES {
        if arg_str == *alias {
            return OsString::from(canonical);
        }
        if let Some(value) = arg_str
            .strip_prefix(alias)
            .and_then(|rest| rest.strip_prefix('='))
        {
            return OsString::from(format!("{canonical}={value}"));
        }
    }

    arg
}
