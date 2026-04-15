use std::fmt::Write as _;
use std::io;

use minijinja::value::{Kwargs, ViaDeserialize};
use minijinja::{Error as MinijinjaError, ErrorKind, Value};
use serde::{Deserialize, Serialize};
use serde_json::ser::Formatter;
use serde_json::{self, Value as JsonValue};

/// Hugging Face-compatible `tojson` filter for chat templates.
///
/// We cannot use MiniJinja's built-in filter directly because HF relies on
/// Python `json.dumps` semantics:
/// - no HTML escaping
/// - extra kwargs such as `ensure_ascii`, `separators`, and `sort_keys`
/// - Python-style `indent` handling
pub(super) fn hf_tojson_filter(
    value: Value,
    kwargs: Kwargs,
) -> std::result::Result<Value, MinijinjaError> {
    let ensure_ascii = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
    let indent = parse_indent(
        kwargs
            .get::<Option<ViaDeserialize<IndentArg>>>("indent")?
            .map(|value| value.0),
    )?;
    let separators = parse_separators(
        kwargs
            .get::<Option<ViaDeserialize<SeparatorsArg>>>("separators")?
            .map(|value| value.0),
        indent.is_some(),
    )?;
    let sort_keys = kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false);

    kwargs.assert_all_used()?;

    let json_value: serde_json::Value = serde_json::to_value(&value).map_err(|e| {
        MinijinjaError::new(
            ErrorKind::InvalidOperation,
            format!("Failed to convert to JSON value: {e}"),
        )
    })?;

    let json_str = {
        let sorted_json;
        let value_to_serialize = if sort_keys {
            sorted_json = sort_json_keys(&json_value);
            &sorted_json
        } else {
            &json_value
        };

        serialize_with_formatter(
            value_to_serialize,
            HfJsonFormatter::new(indent, separators.0, separators.1),
        )?
    };

    let json_str = if ensure_ascii {
        escape_json_non_ascii(&json_str)
    } else {
        json_str
    };

    Ok(Value::from_safe_string(json_str))
}

fn serialize_with_formatter<T: Serialize>(
    value: &T,
    formatter: HfJsonFormatter,
) -> std::result::Result<String, MinijinjaError> {
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, formatter);
    value.serialize(&mut serializer).map_err(|e| {
        MinijinjaError::new(
            ErrorKind::InvalidOperation,
            format!("Failed to serialize JSON: {e}"),
        )
    })?;
    String::from_utf8(buf).map_err(|e| {
        MinijinjaError::new(
            ErrorKind::InvalidOperation,
            format!("Invalid UTF-8 in JSON output: {e}"),
        )
    })
}

#[derive(Deserialize)]
#[serde(untagged)]
enum IndentArg {
    // Python `json.dumps` accepts bool, int, and string indentation styles.
    Bool(bool),
    Integer(i64),
    String(String),
}

fn parse_indent(value: Option<IndentArg>) -> std::result::Result<Option<Vec<u8>>, MinijinjaError> {
    let Some(value) = value else {
        return Ok(None);
    };

    match value {
        IndentArg::Bool(indent) => Ok(Some(if indent { vec![b' '] } else { Vec::new() })),
        IndentArg::Integer(indent) => Ok(Some(if indent > 0 {
            vec![b' '; indent as usize]
        } else {
            Vec::new()
        })),
        IndentArg::String(indent) => Ok(Some(indent.into_bytes())),
    }
}

#[derive(Deserialize)]
struct SeparatorsArg((String, String));

fn parse_separators(
    value: Option<SeparatorsArg>,
    pretty: bool,
) -> std::result::Result<(Vec<u8>, Vec<u8>), MinijinjaError> {
    let default_item_separator = if pretty {
        b",".to_vec()
    } else {
        b", ".to_vec()
    };
    let default_key_separator = b": ".to_vec();

    let Some(value) = value else {
        return Ok((default_item_separator, default_key_separator));
    };

    let SeparatorsArg((item_separator, key_separator)) = value;

    Ok((item_separator.into_bytes(), key_separator.into_bytes()))
}

fn escape_json_non_ascii(json: &str) -> String {
    let mut escaped = String::with_capacity(json.len());

    for ch in json.chars() {
        if ch.is_ascii() {
            escaped.push(ch);
        } else {
            // Match Python's `ensure_ascii=True` behavior by escaping via UTF-16
            // code units, including surrogate pairs for non-BMP characters.
            let mut units = [0; 2];
            for code_unit in ch.encode_utf16(&mut units).iter() {
                let _ = write!(escaped, "\\u{code_unit:04x}");
            }
        }
    }

    escaped
}

/// Formatter that mirrors the subset of `json.dumps` spacing behavior used by
/// HF chat templates, including custom separators in both compact and pretty
/// modes.
struct HfJsonFormatter {
    current_indent: usize,
    has_value: bool,
    indent: Option<Vec<u8>>,
    item_separator: Vec<u8>,
    key_separator: Vec<u8>,
}

impl HfJsonFormatter {
    fn new(indent: Option<Vec<u8>>, item_separator: Vec<u8>, key_separator: Vec<u8>) -> Self {
        Self {
            current_indent: 0,
            has_value: false,
            indent,
            item_separator,
            key_separator,
        }
    }

    fn is_pretty(&self) -> bool {
        self.indent.is_some()
    }

    fn write_indent<W>(&self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if let Some(indent) = &self.indent {
            for _ in 0..self.current_indent {
                writer.write_all(indent)?;
            }
        }
        Ok(())
    }
}

impl Formatter for HfJsonFormatter {
    fn begin_array<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if self.is_pretty() {
            self.current_indent += 1;
            self.has_value = false;
        }
        writer.write_all(b"[")
    }

    fn end_array<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if self.is_pretty() {
            self.current_indent -= 1;
            if self.has_value {
                writer.write_all(b"\n")?;
                self.write_indent(writer)?;
            }
        }
        writer.write_all(b"]")
    }

    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if self.is_pretty() {
            if first {
                writer.write_all(b"\n")?;
            } else {
                writer.write_all(&self.item_separator)?;
                writer.write_all(b"\n")?;
            }
            self.write_indent(writer)
        } else if first {
            Ok(())
        } else {
            writer.write_all(&self.item_separator)
        }
    }

    fn end_array_value<W>(&mut self, _writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if self.is_pretty() {
            self.has_value = true;
        }
        Ok(())
    }

    fn begin_object<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if self.is_pretty() {
            self.current_indent += 1;
            self.has_value = false;
        }
        writer.write_all(b"{")
    }

    fn end_object<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if self.is_pretty() {
            self.current_indent -= 1;
            if self.has_value {
                writer.write_all(b"\n")?;
                self.write_indent(writer)?;
            }
        }
        writer.write_all(b"}")
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if self.is_pretty() {
            if first {
                writer.write_all(b"\n")?;
            } else {
                writer.write_all(&self.item_separator)?;
                writer.write_all(b"\n")?;
            }
            self.write_indent(writer)
        } else if first {
            Ok(())
        } else {
            writer.write_all(&self.item_separator)
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(&self.key_separator)
    }

    fn end_object_value<W>(&mut self, _writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if self.is_pretty() {
            self.has_value = true;
        }
        Ok(())
    }
}

/// Recursively sort all object keys in a JSON value.
fn sort_json_keys(value: &JsonValue) -> JsonValue {
    match value {
        JsonValue::Object(map) => {
            let mut sorted: serde_json::Map<String, JsonValue> = serde_json::Map::new();
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            for key in keys {
                sorted.insert(key.clone(), sort_json_keys(&map[key]));
            }
            JsonValue::Object(sorted)
        }
        JsonValue::Array(arr) => JsonValue::Array(arr.iter().map(sort_json_keys).collect()),
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use minijinja::Environment;
    use serde_json::json;

    use super::hf_tojson_filter;

    fn render(template: &str, payload: serde_json::Value) -> String {
        let mut env = Environment::new();
        env.add_filter("tojson", hf_tojson_filter);
        env.render_str(template, json!({ "payload": payload }))
            .unwrap()
    }

    #[test]
    fn tojson_does_not_html_escape_like_minijinja_builtin() {
        let rendered = render("{{ payload|tojson }}", json!("<tag>&'"));
        assert_eq!(rendered, "\"<tag>&'\"");
    }

    #[test]
    fn tojson_supports_sort_keys_recursively() {
        let rendered = render(
            "{{ payload|tojson(sort_keys=true) }}",
            json!({
                "z": {"b": 1, "a": 2},
                "a": 0
            }),
        );

        assert_eq!(rendered, "{\"a\": 0, \"z\": {\"a\": 2, \"b\": 1}}");
    }

    #[test]
    fn tojson_supports_indent() {
        let rendered = render("{{ payload|tojson(indent=2) }}", json!([1, 2]));

        assert_eq!(rendered, "[\n  1,\n  2\n]");
    }

    #[test]
    fn tojson_supports_ensure_ascii_false() {
        let rendered = render("{{ payload|tojson(ensure_ascii=false) }}", json!("中文"));
        assert_eq!(rendered, "\"中文\"");
    }

    #[test]
    fn tojson_supports_ensure_ascii_true() {
        let rendered = render("{{ payload|tojson(ensure_ascii=true) }}", json!("中文"));
        assert_eq!(rendered, "\"\\u4e2d\\u6587\"");
    }

    #[test]
    fn tojson_supports_separators() {
        let rendered = render(
            "{{ payload|tojson(separators=[',', ':']) }}",
            json!({
                "x": [1, 2]
            }),
        );

        assert_eq!(rendered, "{\"x\":[1,2]}");
    }

    #[test]
    fn tojson_supports_negative_indent_as_newline_only() {
        let rendered = render("{{ payload|tojson(indent=-1) }}", json!([1, 2]));
        assert_eq!(rendered, "[\n1,\n2\n]");
    }

    #[test]
    fn tojson_supports_string_indent() {
        let rendered = render("{{ payload|tojson(indent='  ') }}", json!([1, 2]));
        assert_eq!(rendered, "[\n  1,\n  2\n]");
    }

    #[test]
    fn tojson_supports_boolean_indent() {
        let rendered_true = render("{{ payload|tojson(indent=true) }}", json!([1, 2]));
        assert_eq!(rendered_true, "[\n 1,\n 2\n]");

        let rendered_false = render("{{ payload|tojson(indent=false) }}", json!([1, 2]));
        assert_eq!(rendered_false, "[\n1,\n2\n]");
    }

    #[test]
    fn tojson_combines_indent_sort_keys_separators_and_ensure_ascii() {
        let rendered = render(
            "{{ payload|tojson(ensure_ascii=true, sort_keys=true, separators=[',', ':'], indent='  ') }}",
            json!({
                "b": "<中>",
                "a": [1, 2]
            }),
        );

        assert_eq!(
            rendered,
            "{\n  \"a\":[\n    1,\n    2\n  ],\n  \"b\":\"<\\u4e2d>\"\n}"
        );
    }
}
