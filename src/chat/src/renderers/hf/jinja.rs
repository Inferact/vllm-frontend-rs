//! Chat template support for tokenizers using Jinja2 templates.
//!
//! This module is inlined from SMG's tokenizer crate with local adaptations:
//! - thinking-related detection/state is removed
//! - special tokens are wired to `vllm_text::backends::hf::HfSpecialTokens`

use std::collections::HashMap;
use std::fs;

use anyhow::{Result, anyhow};
use minijinja::machinery::ast::{Expr, Stmt};
use minijinja::machinery::{WhitespaceConfig, parse};
use minijinja::syntax::SyntaxConfig;
use minijinja::value::Kwargs;
use minijinja::{Environment, Error as MinijinjaError, ErrorKind, Value, context};
use serde::Serialize;
use serde_json::ser::PrettyFormatter;
use serde_json::{self, Value as JsonValue};
use vllm_text::backends::hf::HfSpecialTokens;

/// Chat template content format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatTemplateContentFormat {
    /// Content is a simple string.
    #[default]
    String,
    /// Content is a list of structured parts (OpenAI format).
    OpenAI,
}

impl std::fmt::Display for ChatTemplateContentFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String => write!(f, "string"),
            Self::OpenAI => write!(f, "openai"),
        }
    }
}

/// Detect the content format expected by a Jinja2 chat template.
pub fn detect_chat_template_content_format(template: &str) -> ChatTemplateContentFormat {
    detect_format_with_ast(template)
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
            Stmt::Set(s) => {
                if Self::is_var_access(&s.target, "content")
                    && self.is_any_scope_var_content(&s.expr)
                {
                    self.flags.saw_assignment = true;
                }
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
            Expr::GetItem(gi) => {
                if (matches!(&gi.expr, Expr::Var(v) if v.id == "content")
                    || self.is_any_scope_var_content(&gi.expr))
                    && Self::is_numeric_const(&gi.subscript_expr)
                {
                    self.flags.saw_structure = true;
                }
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
        ChatTemplateContentFormat::OpenAI
    } else {
        ChatTemplateContentFormat::String
    }
}

/// Parameters for chat template application.
#[derive(Default)]
pub struct ChatTemplateParams<'a> {
    pub add_generation_prompt: bool,
    pub tools: Option<&'a [serde_json::Value]>,
    pub documents: Option<&'a [serde_json::Value]>,
    pub template_kwargs: Option<&'a HashMap<String, serde_json::Value>>,
    /// Special tokens to inject into the template context.
    pub special_tokens: Option<&'a HfSpecialTokens>,
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

    env.add_template_owned("chat".to_owned(), template)
        .map_err(|e| anyhow!("Failed to add template: {e}"))?;

    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_filter("tojson", tojson_filter);

    Ok(env)
}

/// Convert an optional token string to a minijinja value.
fn special_token_value(token: Option<&str>) -> Value {
    token.map_or(Value::UNDEFINED, Value::from)
}

fn render_chat_template(
    env: &Environment<'_>,
    messages: &[serde_json::Value],
    params: ChatTemplateParams,
) -> Result<String> {
    let tmpl = env
        .get_template("chat")
        .map_err(|e| anyhow!("Failed to get template: {e}"))?;

    let minijinja_messages: Vec<Value> = messages.iter().map(Value::from_serialize).collect();
    let tools_value = params.tools.map_or(Value::UNDEFINED, Value::from_serialize);
    let documents_value = params
        .documents
        .map_or(Value::UNDEFINED, Value::from_serialize);

    let bos_value = special_token_value(
        params
            .special_tokens
            .and_then(|st| st.bos_token.as_ref())
            .map(|token| token.as_str()),
    );
    let eos_value = special_token_value(
        params
            .special_tokens
            .and_then(|st| st.eos_token.as_ref())
            .map(|token| token.as_str()),
    );
    let unk_value = special_token_value(
        params
            .special_tokens
            .and_then(|st| st.unk_token.as_ref())
            .map(|token| token.as_str()),
    );
    let pad_value = special_token_value(
        params
            .special_tokens
            .and_then(|st| st.pad_token.as_ref())
            .map(|token| token.as_str()),
    );

    let base_context = context! {
        messages => &minijinja_messages,
        add_generation_prompt => params.add_generation_prompt,
        tools => tools_value,
        documents => documents_value,
        bos_token => bos_value,
        eos_token => eos_value,
        unk_token => unk_value,
        pad_token => pad_value,
    };

    let ctx = if let Some(kwargs) = params.template_kwargs {
        context! {
            ..base_context,
            ..Value::from_serialize(kwargs)
        }
    } else {
        base_context
    };

    tmpl.render(&ctx)
        .map_err(|e| anyhow!("Failed to render template: {e}"))
}

/// Load chat template from a file (`.jinja` or `.json` containing Jinja).
pub fn load_chat_template_from_file(template_path: &str) -> Result<Option<String>> {
    let content = fs::read_to_string(template_path)
        .map_err(|e| anyhow!("Failed to read chat template file: {e}"))?;

    if template_path.ends_with(".json") {
        let json_value: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| anyhow!("Failed to parse chat_template.json: {e}"))?;

        if let Some(template_str) = json_value.as_str() {
            return Ok(Some(template_str.to_string()));
        } else if let Some(obj) = json_value.as_object()
            && let Some(template_value) = obj.get("chat_template")
            && let Some(template_str) = template_value.as_str()
        {
            return Ok(Some(template_str.to_string()));
        }

        return Err(anyhow!(
            "chat_template.json does not contain a valid template",
        ));
    }

    let template = content.trim().replace("\\n", "\n");
    Ok(Some(template))
}

/// One compiled chat template with its Jinja environment and detected content format.
pub struct CompiledChatTemplate {
    /// Cached, fully-configured environment for one compiled template.
    env: Environment<'static>,
    content_format: ChatTemplateContentFormat,
}

impl std::fmt::Debug for CompiledChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompiledChatTemplate")
            .field("content_format", &self.content_format)
            .finish()
    }
}

impl CompiledChatTemplate {
    pub fn new(template: String) -> Result<Self> {
        let content_format = detect_chat_template_content_format(&template);
        let env = build_environment(template)?;
        Ok(Self {
            env,
            content_format,
        })
    }

    pub fn apply(
        &self,
        messages: &[serde_json::Value],
        params: ChatTemplateParams,
    ) -> Result<String> {
        render_chat_template(&self.env, messages, params)
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
        let result = template.apply(&[], ChatTemplateParams::default()).unwrap();
        assert_eq!(result, "[]");
    }

    #[test]
    fn test_chat_template_state_invalid_template() {
        let result = CompiledChatTemplate::new("{% invalid".to_string());
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Failed to add template"),
            "Error should explain parse failure, got: {err}"
        );
    }

    #[test]
    fn test_special_tokens_injected_into_context() {
        let template = "{{ bos_token }}{% for message in messages %}{{ message.content }}{% endfor %}{{ eos_token }}";
        let template = CompiledChatTemplate::new(template.to_string()).unwrap();

        let messages = vec![serde_json::json!({"role": "user", "content": "hello"})];
        let special_tokens = HfSpecialTokens {
            bos_token: Some(NamedSpecialToken::Text("<s>".to_string())),
            eos_token: Some(NamedSpecialToken::Text("</s>".to_string())),
            ..Default::default()
        };

        let result = template
            .apply(
                &messages,
                ChatTemplateParams {
                    special_tokens: Some(&special_tokens),
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(result, "<s>hello</s>");
    }

    #[test]
    fn test_special_tokens_undefined_when_not_provided() {
        let template = "{% if bos_token is defined %}{{ bos_token }}{% endif %}hello";
        let template = CompiledChatTemplate::new(template.to_string()).unwrap();

        let result = template.apply(&[], ChatTemplateParams::default()).unwrap();
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
            .apply(
                &[],
                ChatTemplateParams {
                    special_tokens: Some(&special_tokens),
                    ..Default::default()
                },
            )
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
            .apply(
                &[],
                ChatTemplateParams {
                    template_kwargs: Some(&kwargs),
                    ..Default::default()
                },
            )
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
