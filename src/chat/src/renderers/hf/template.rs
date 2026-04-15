//! Chat template support for tokenizers using Jinja2 templates.
//!
//! This module is inlined from SMG's tokenizer crate with local adaptations:
//! - thinking-related detection/state is removed
//! - special tokens are wired to `vllm_text::backends::hf::HfSpecialTokens`

use std::collections::HashMap;
use std::fs;

use minijinja::machinery::ast::{Expr, Stmt};
use minijinja::machinery::{WhitespaceConfig, parse};
use minijinja::syntax::SyntaxConfig;
use minijinja::value::Kwargs;
use minijinja::{Environment, Error as MinijinjaError, ErrorKind, Value};
use serde::{Deserialize, Serialize};
use serde_json::ser::PrettyFormatter;
use serde_json::{self, Value as JsonValue};
use vllm_text::backends::hf::HfSpecialTokens;

use super::error::TemplateError;
use crate::renderers::hf::{TemplateMessage, TemplateTool};

type Result<T> = std::result::Result<T, TemplateError>;

/// Chat template content format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplateContentFormat {
    /// Content is a simple string.
    #[default]
    String,
    /// Content is a list of structured parts (OpenAI format).
    OpenAi,
}

/// Flags tracking which OpenAI-style patterns we've seen.
#[derive(Default, Debug, Clone, Copy)]
struct Flags {
    saw_iteration: bool,
    saw_structure: bool,
    saw_assignment: bool,
    saw_macro: bool,
}

impl Flags {
    fn any(self) -> bool {
        // `saw_assignment` alone (e.g. `set content = message.content`) is not sufficient to
        // classify as OpenAI format. Many string-format templates use this pattern to extract
        // content into a local variable, then check `content is string`.
        self.saw_iteration || self.saw_structure || self.saw_macro
    }
}

/// Single-pass AST detector with scope tracking.
struct Detector<'a> {
    ast: &'a Stmt<'a>,
    /// Message loop vars currently in scope (e.g., `message`, `m`, `msg`).
    scope: std::collections::VecDeque<String>,
    scope_set: std::collections::HashSet<String>,
    flags: Flags,
}

impl<'a> Detector<'a> {
    fn new(ast: &'a Stmt<'a>) -> Self {
        Self {
            ast,
            scope: std::collections::VecDeque::new(),
            scope_set: std::collections::HashSet::new(),
            flags: Flags::default(),
        }
    }

    fn run(mut self) -> Flags {
        self.walk_stmt(self.ast);
        self.flags
    }

    fn push_scope(&mut self, var: String) {
        self.scope.push_back(var.clone());
        self.scope_set.insert(var);
    }

    fn pop_scope(&mut self) {
        if let Some(v) = self.scope.pop_back() {
            self.scope_set.remove(&v);
        }
    }

    fn is_var_access(expr: &Expr, varname: &str) -> bool {
        matches!(expr, Expr::Var(v) if v.id == varname)
    }

    fn is_const_str(expr: &Expr, value: &str) -> bool {
        matches!(expr, Expr::Const(c) if c.value.as_str() == Some(value))
    }

    fn is_numeric_const(expr: &Expr) -> bool {
        matches!(expr, Expr::Const(c) if c.value.is_number())
    }

    /// Check if expr is varname.content or varname["content"].
    fn is_var_dot_content(expr: &Expr, varname: &str) -> bool {
        match expr {
            Expr::GetAttr(g) => Self::is_var_access(&g.expr, varname) && g.name == "content",
            Expr::GetItem(g) => {
                Self::is_var_access(&g.expr, varname)
                    && Self::is_const_str(&g.subscript_expr, "content")
            }
            Expr::Filter(f) => f
                .expr
                .as_ref()
                .is_some_and(|e| Self::is_var_dot_content(e, varname)),
            Expr::Test(t) => Self::is_var_dot_content(&t.expr, varname),
            _ => false,
        }
    }

    /// Check if expr accesses `.content` on any variable in scope, or any descendant of it.
    fn is_any_scope_var_content(&self, expr: &Expr) -> bool {
        let mut current_expr = expr;
        loop {
            if self
                .scope_set
                .iter()
                .any(|v| Self::is_var_dot_content(current_expr, v))
            {
                return true;
            }
            match current_expr {
                Expr::GetAttr(g) => current_expr = &g.expr,
                Expr::GetItem(g) => current_expr = &g.expr,
                _ => return false,
            }
        }
    }

    fn walk_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Template(t) => {
                for ch in &t.children {
                    self.walk_stmt(ch);
                }
            }
            Stmt::ForLoop(fl) => {
                if let Expr::Var(iter) = &fl.iter
                    && iter.id == "messages"
                    && let Expr::Var(target) = &fl.target
                {
                    self.push_scope(target.id.to_string());
                }

                if self.is_any_scope_var_content(&fl.iter) {
                    self.flags.saw_iteration = true;
                }
                if matches!(&fl.iter, Expr::Var(v) if v.id == "content") {
                    self.flags.saw_iteration = true;
                }

                for b in &fl.body {
                    self.walk_stmt(b);
                }

                if let Expr::Var(iter) = &fl.iter
                    && iter.id == "messages"
                    && matches!(&fl.target, Expr::Var(_))
                {
                    self.pop_scope();
                }
            }
            Stmt::IfCond(ic) => {
                self.inspect_expr_for_structure(&ic.expr);

                for b in &ic.true_body {
                    self.walk_stmt(b);
                }
                for b in &ic.false_body {
                    self.walk_stmt(b);
                }
            }
            Stmt::EmitExpr(e) => {
                self.inspect_expr_for_structure(&e.expr);
            }
            Stmt::Set(s)
                if Self::is_var_access(&s.target, "content")
                    && self.is_any_scope_var_content(&s.expr) =>
            {
                self.flags.saw_assignment = true;
            }
            Stmt::Macro(m) => {
                let mut has_type_check = false;
                let mut has_loop = false;
                Self::scan_macro_body(&m.body, &mut has_type_check, &mut has_loop);
                if has_type_check && has_loop {
                    self.flags.saw_macro = true;
                }
            }
            _ => {}
        }
    }

    fn inspect_expr_for_structure(&mut self, expr: &Expr) {
        if self.flags.saw_structure {
            return;
        }

        match expr {
            Expr::GetItem(gi)
                if (matches!(&gi.expr, Expr::Var(v) if v.id == "content")
                    || self.is_any_scope_var_content(&gi.expr))
                    && Self::is_numeric_const(&gi.subscript_expr) =>
            {
                self.flags.saw_structure = true;
            }
            Expr::Filter(f) => {
                if f.name == "length" {
                    if let Some(inner) = &f.expr {
                        let inner_ref: &Expr = inner;
                        let is_content_var = matches!(inner_ref, Expr::Var(v) if v.id == "content");
                        if is_content_var || self.is_any_scope_var_content(inner_ref) {
                            self.flags.saw_structure = true;
                        }
                    }
                } else if let Some(inner) = &f.expr {
                    let inner_ref: &Expr = inner;
                    self.inspect_expr_for_structure(inner_ref);
                }
            }
            Expr::Test(t) => self.inspect_expr_for_structure(&t.expr),
            Expr::GetAttr(g) => {
                self.inspect_expr_for_structure(&g.expr);
            }
            Expr::BinOp(op) => {
                self.inspect_expr_for_structure(&op.left);
                self.inspect_expr_for_structure(&op.right);
            }
            Expr::UnaryOp(op) => {
                self.inspect_expr_for_structure(&op.expr);
            }
            _ => {}
        }
    }

    fn scan_macro_body(body: &[Stmt], has_type_check: &mut bool, has_loop: &mut bool) {
        for s in body {
            if *has_type_check && *has_loop {
                return;
            }

            match s {
                Stmt::IfCond(ic) => {
                    if matches!(&ic.expr, Expr::Test(_)) {
                        *has_type_check = true;
                    }
                    Self::scan_macro_body(&ic.true_body, has_type_check, has_loop);
                    Self::scan_macro_body(&ic.false_body, has_type_check, has_loop);
                }
                Stmt::ForLoop(fl) => {
                    *has_loop = true;
                    Self::scan_macro_body(&fl.body, has_type_check, has_loop);
                }
                Stmt::Template(t) => {
                    Self::scan_macro_body(&t.children, has_type_check, has_loop);
                }
                _ => {}
            }
        }
    }
}

/// AST-based detection using minijinja's unstable machinery.
fn detect_format_with_ast(template: &str) -> ChatTemplateContentFormat {
    let ast = match parse(
        template,
        "template",
        SyntaxConfig {},
        WhitespaceConfig::default(),
    ) {
        Ok(ast) => ast,
        Err(_) => return ChatTemplateContentFormat::String,
    };

    let flags = Detector::new(&ast).run();
    if flags.any() {
        ChatTemplateContentFormat::OpenAi
    } else {
        ChatTemplateContentFormat::String
    }
}

/// Detect the content format expected by a Jinja2 chat template.
pub fn detect_chat_template_content_format(template: &str) -> ChatTemplateContentFormat {
    detect_format_with_ast(template)
}

/// Custom `tojson` filter compatible with HuggingFace transformers' implementation.
fn tojson_filter(value: Value, kwargs: Kwargs) -> std::result::Result<Value, MinijinjaError> {
    let _ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let indent: Option<i64> = kwargs.get("indent")?;
    let _separators: Option<Value> = kwargs.get("separators")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;

    kwargs.assert_all_used()?;

    let json_value: serde_json::Value = serde_json::to_value(&value).map_err(|e| {
        MinijinjaError::new(
            ErrorKind::InvalidOperation,
            format!("Failed to convert to JSON value: {e}"),
        )
    })?;

    fn serialize_with_indent<T: Serialize>(
        value: &T,
        spaces: usize,
    ) -> std::result::Result<String, MinijinjaError> {
        let indent_str = vec![b' '; spaces];
        let formatter = PrettyFormatter::with_indent(&indent_str);
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

    let json_str: std::result::Result<String, MinijinjaError> = {
        let sorted_json;
        let value_to_serialize = if sort_keys.unwrap_or(false) {
            sorted_json = sort_json_keys(&json_value);
            &sorted_json
        } else {
            &json_value
        };

        if let Some(spaces) = indent {
            if spaces < 0 {
                return Err(MinijinjaError::new(
                    ErrorKind::InvalidOperation,
                    "indent cannot be negative",
                ));
            }
            serialize_with_indent(value_to_serialize, spaces as usize)
        } else {
            serde_json::to_string(value_to_serialize).map_err(|e| {
                MinijinjaError::new(
                    ErrorKind::InvalidOperation,
                    format!("Failed to serialize JSON: {e}"),
                )
            })
        }
    };

    json_str.map(Value::from_safe_string)
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

/// Build a pre-configured environment with the given template string.
fn build_environment(template: String) -> Result<Environment<'static>> {
    let mut env = Environment::new();

    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);

    env.add_template_owned("chat".to_owned(), template)?;

    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_filter("tojson", tojson_filter);

    Ok(env)
}

#[serde_with::skip_serializing_none]
#[derive(Default, Serialize)]
pub(super) struct TemplateContext<'a> {
    pub(super) messages: &'a [TemplateMessage],
    pub(super) add_generation_prompt: bool,
    pub(super) tools: Option<&'a [TemplateTool]>,
    pub(super) documents: Option<&'a [serde_json::Value]>,
    #[serde(flatten)]
    pub(super) special_tokens: Option<&'a HfSpecialTokens>,
    #[serde(flatten)]
    pub(super) template_kwargs: Option<&'a HashMap<String, serde_json::Value>>,
}

/// Load chat template from a file (`.jinja` or `.json` containing Jinja).
pub fn load_chat_template_from_file(template_path: &str) -> Result<Option<String>> {
    let content = fs::read_to_string(template_path).map_err(TemplateError::ReadTemplateFile)?;

    if template_path.ends_with(".json") {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum ChatTemplateFile {
            String(String),
            Object { chat_template: String },
        }

        let json_value =
            serde_json::from_str(&content).map_err(TemplateError::ParseTemplateJson)?;
        let json_template =
            serde_json::from_value(json_value).map_err(|_| TemplateError::InvalidTemplateJson)?;

        return Ok(Some(match json_template {
            ChatTemplateFile::String(template) => template,
            ChatTemplateFile::Object { chat_template } => chat_template,
        }));
    }

    let template = content.trim().replace("\\n", "\n");
    Ok(Some(template))
}

/// One compiled chat template with its Jinja environment and detected content format.
pub(super) struct CompiledChatTemplate {
    /// Cached, fully-configured environment for one compiled template.
    env: Environment<'static>,
    content_format: ChatTemplateContentFormat,
}

impl CompiledChatTemplate {
    /// Compile the given chat template string into a [`CompiledChatTemplate`].
    pub fn new(template: String) -> Result<Self> {
        let content_format = detect_chat_template_content_format(&template);
        let env = build_environment(template)?;
        Ok(Self {
            env,
            content_format,
        })
    }

    /// Apply the compiled template to the given context and return the rendered prompt.
    pub fn apply(&self, ctx: TemplateContext<'_>) -> Result<String> {
        let tmpl = self.env.get_template("chat")?;
        tmpl.render(ctx).map_err(TemplateError::from)
    }

    pub fn content_format(&self) -> ChatTemplateContentFormat {
        self.content_format
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;
    use vllm_text::backends::hf::{HfSpecialTokens, NamedSpecialToken};

    use super::*;

    #[test]
    fn test_chat_template_state_valid_template() {
        let template = CompiledChatTemplate::new("{{ messages }}".to_string()).unwrap();
        assert_eq!(template.content_format(), ChatTemplateContentFormat::String);
        let result = template.apply(TemplateContext::default()).unwrap();
        assert_eq!(result, "[]");
    }

    #[test]
    fn test_chat_template_state_invalid_template() {
        let result = CompiledChatTemplate::new("{% invalid".to_string());
        assert!(result.is_err());
        let err = result.err().unwrap().to_string();
        assert!(
            err.contains("failed to render jinja template"),
            "Error should explain parse failure, got: {err}"
        );
    }

    #[test]
    fn test_special_tokens_injected_into_context() {
        let template = "{{ bos_token }}hello{{ eos_token }}";
        let template = CompiledChatTemplate::new(template.to_string()).unwrap();

        let special_tokens = HfSpecialTokens {
            bos_token: Some(NamedSpecialToken::Text("<s>".to_string())),
            eos_token: Some(NamedSpecialToken::Text("</s>".to_string())),
            ..Default::default()
        };

        let result = template
            .apply(TemplateContext {
                special_tokens: Some(&special_tokens),
                ..Default::default()
            })
            .unwrap();

        assert_eq!(result, "<s>hello</s>");
    }

    #[test]
    fn test_special_tokens_undefined_when_not_provided() {
        let template = "{% if bos_token is defined %}{{ bos_token }}{% endif %}hello";
        let template = CompiledChatTemplate::new(template.to_string()).unwrap();

        let result = template.apply(TemplateContext::default()).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_special_tokens_partial() {
        let template =
            "{{ bos_token }}hello{% if eos_token is defined %}{{ eos_token }}{% endif %}";
        let template = CompiledChatTemplate::new(template.to_string()).unwrap();

        let special_tokens = HfSpecialTokens {
            bos_token: Some(NamedSpecialToken::Text("<s>".to_string())),
            eos_token: None,
            ..Default::default()
        };

        let result = template
            .apply(TemplateContext {
                special_tokens: Some(&special_tokens),
                ..Default::default()
            })
            .unwrap();

        assert_eq!(result, "<s>hello");
    }

    #[test]
    fn test_tojson_filter_supports_indent_and_sort_keys() {
        let template = CompiledChatTemplate::new(
            "{{ payload | tojson(indent=2, sort_keys=true) }}".to_string(),
        )
        .unwrap();
        let mut kwargs = HashMap::new();
        kwargs.insert("payload".to_string(), serde_json::json!({"b": 1, "a": 2}));

        let result = template
            .apply(TemplateContext {
                template_kwargs: Some(&kwargs),
                ..Default::default()
            })
            .unwrap();

        assert_eq!(result, "{\n  \"a\": 2,\n  \"b\": 1\n}");
    }

    #[test]
    fn test_load_chat_template_from_file_jinja() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("chat_template.jinja");
        fs::write(&path, "{{ messages }}").unwrap();

        let template = load_chat_template_from_file(path.to_str().unwrap()).unwrap();

        assert_eq!(template.as_deref(), Some("{{ messages }}"));
    }

    #[test]
    fn test_load_chat_template_from_file_json_string() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("chat_template.json");
        fs::write(&path, "\"{{ messages }}\"").unwrap();

        let template = load_chat_template_from_file(path.to_str().unwrap()).unwrap();

        assert_eq!(template.as_deref(), Some("{{ messages }}"));
    }

    #[test]
    fn test_load_chat_template_from_file_json_object() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("chat_template.json");
        fs::write(&path, r#"{"chat_template":"{{ messages }}"}"#).unwrap();

        let template = load_chat_template_from_file(path.to_str().unwrap()).unwrap();

        assert_eq!(template.as_deref(), Some("{{ messages }}"));
    }
}
