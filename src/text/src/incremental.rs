use std::mem::take;

use unicode_segmentation::UnicodeSegmentation;

use crate::backend::TextBackend;
use crate::error::Result;

/// Stateful incremental decoder that emits text chunks one token at a time.
pub trait IncrementalDecoder: Send {
    /// Push one generated token and return the newly decoded text chunk, if any.
    ///
    /// Returns `Ok((string, num_bytes_added))`, where string is `None` when the
    /// token does not yet produce a stable text fragment (e.g. in the middle of a
    /// multi-byte UTF-8 sequence, incomplete grapheme, or due to stop-string buffering).
    fn push_token(&mut self, token_id: u32) -> Result<(Option<String>, usize)>;

    /// Flush any remaining buffered text that has not yet been emitted.
    ///
    /// Called after the final generated token to force out incomplete fragments.
    fn flush(&mut self) -> Result<Option<String>>;

    /// Return cumulative decoded text so far.
    fn output(&self) -> &str;
    fn take_output(& mut self) -> String;

    /// Return number of bytes currently held back.
    fn held_back_count(&self) -> usize;
}

/// [`IncrementalDecoder`] built on [`TextBackend::decode()`] with prefix-diffing.
///
/// This is the same sliding-window algorithm used by `tokenizers::DecodeStream` and
/// `fastokens::DecodeStream`.
pub(crate) struct DecodeStream<'a, B: TextBackend + ?Sized> {
    backend: &'a B,
    skip_special_tokens: bool,
    hold_back_bytes: usize,
    ids: Vec<u32>,
    prefix: String,
    prefix_index: usize,
    str_buffer: String,
    cumulative_output: String,
}

impl<'a, B: TextBackend + ?Sized> DecodeStream<'a, B> {
    pub(crate) fn new(
        backend: &'a B,
        prompt_token_ids: &[u32],
        skip_special_tokens: bool,
        hold_back_bytes: usize,
    ) -> Self {
        Self {
            backend,
            skip_special_tokens,
            hold_back_bytes,
            ids: prompt_token_ids.to_vec(),
            prefix: String::new(),
            prefix_index: 0,
            str_buffer: String::new(),
            cumulative_output: String::new(),
        }
    }
}

impl<B: TextBackend + ?Sized> IncrementalDecoder for DecodeStream<'_, B> {
    fn push_token(&mut self, token_id: u32) -> Result<(Option<String>, usize)> {
        if self.prefix.is_empty() && !self.ids.is_empty() {
            let new_prefix = self.backend.decode(&self.ids, self.skip_special_tokens)?;
            if !new_prefix.ends_with('\u{FFFD}') {
                self.prefix = new_prefix;
                self.prefix_index = self.ids.len();
            }
        }

        self.ids.push(token_id);
        let mut string = self.backend.decode(&self.ids, self.skip_special_tokens)?;
        let prefix_len = self.prefix.len();
        let mut added_bytes = 0;
        if string.len() > prefix_len && !string.ends_with('\u{FFFD}') {
            let new_text = &string[prefix_len..];
            added_bytes = new_text.len();
            self.cumulative_output.push_str(new_text);
            if self.str_buffer.is_empty() {
                string.replace_range(..prefix_len, "");
                self.str_buffer = string
            } else {
                self.str_buffer.push_str(new_text);
            }

            self.ids.drain(..self.prefix_index);
            self.prefix = self.backend.decode(&self.ids, self.skip_special_tokens)?;
            self.prefix_index = self.ids.len();

            if self.hold_back_bytes == 0 {
                // No buffering needed — emit everything immediately.
                return Ok((Some(take(&mut self.str_buffer)), added_bytes));
            }
            // Keep at least hold_back_bytes in the str_buffer
            let cutoff = self.str_buffer.len().saturating_sub(self.hold_back_bytes + added_bytes);
            if cutoff != 0 {
                // Ensure that we return full grapheme clusters
                for (idx, _) in self.str_buffer.grapheme_indices(true).rev() {
                    if idx <= cutoff && idx > 0 {
                        return Ok((Some(self.str_buffer.drain(..idx).collect()), added_bytes));
                    }
                }
            }
        }
        Ok((None, added_bytes))
    }

    fn flush(&mut self) -> Result<Option<String>> {
        let text = if self.ids.is_empty() {
            take(&mut self.str_buffer)
        } else {
            let mut text = self.backend.decode(&self.ids, self.skip_special_tokens)?;
            let prefix_len = self.prefix.len();
            self.ids.clear();
            self.prefix.clear();
            self.prefix_index = 0;
            let new_text = &text[prefix_len..];
            self.cumulative_output.push_str(new_text);
            if self.str_buffer.is_empty() {
                text.replace_range(..prefix_len, "");
                text
            } else {
                self.str_buffer.push_str(new_text);
                take(&mut self.str_buffer)
            }
        };
        Ok((!text.is_empty()).then_some(text))
    }

    fn output(&self) -> &str {
        &self.cumulative_output
    }

    fn held_back_count(&self) -> usize {
        self.str_buffer.len()
    }

    fn take_output(& mut self) -> String {
        take(&mut self.cumulative_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Backend that treats each token ID as a raw byte, producing lossy UTF-8.
    #[derive(Debug)]
    struct Utf8Backend;

    impl TextBackend for Utf8Backend {
        fn encode(&self, _text: &str) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn decode(&self, token_ids: &[u32], _skip_special_tokens: bool) -> Result<String> {
            let bytes = token_ids.iter().map(|id| *id as u8).collect::<Vec<_>>();
            Ok(String::from_utf8_lossy(&bytes).into_owned())
        }
    }

    #[test]
    fn holds_incomplete_utf8_until_complete() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        // 你 = U+4F60 = 0xE4 0xBD 0xA0
        assert_eq!(decoder.push_token(0xe4).unwrap().0, None);
        assert_eq!(decoder.push_token(0xbd).unwrap().0, None);
        assert_eq!(decoder.push_token(0xa0).unwrap().0.as_deref(), Some("你"));
    }

    #[test]
    fn emits_ascii_immediately() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        assert_eq!(decoder.push_token(b'o' as u32).unwrap().0.as_deref(), Some("o"));
        assert_eq!(decoder.push_token(b'k' as u32).unwrap().0.as_deref(), Some("k"));
    }

    #[test]
    fn flush_returns_none_when_fully_consumed() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        assert_eq!(decoder.push_token(b'o' as u32).unwrap().0.as_deref(), Some("o"));
        assert_eq!(decoder.push_token(b'k' as u32).unwrap().0.as_deref(), Some("k"));
        assert_eq!(decoder.flush().unwrap(), None);
    }

    #[test]
    fn flush_emits_buffered_incomplete_utf8() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        // Push incomplete multi-byte sequence — step returns None.
        assert_eq!(decoder.push_token(0xe4).unwrap().0, None);
        assert_eq!(decoder.push_token(0xbd).unwrap().0, None);

        // Flush forces out whatever the decoder can produce (lossy replacement).
        let flushed = decoder.flush().unwrap();
        assert!(flushed.is_some());
    }

    /// Backend where token 0 is a special token.
    #[derive(Debug)]
    struct SpecialTokenBackend;

    impl TextBackend for SpecialTokenBackend {
        fn encode(&self, _text: &str) -> Result<Vec<u32>> {
            unreachable!()
        }

        fn decode(&self, token_ids: &[u32], skip_special_tokens: bool) -> Result<String> {
            let mut text = String::new();
            for &token_id in token_ids {
                match token_id {
                    0 if !skip_special_tokens => text.push_str("<special>"),
                    0 => {}
                    1 => text.push('a'),
                    _ => {}
                }
            }
            Ok(text)
        }
    }

    #[test]
    fn respects_skip_special_tokens() {
        let backend = SpecialTokenBackend;
        let mut skip_decoder = backend.create_decode_stream(&[], true, 0);
        let mut keep_decoder = backend.create_decode_stream(&[], false, 0);

        assert_eq!(skip_decoder.push_token(0).unwrap().0, None);
        assert_eq!(keep_decoder.push_token(0).unwrap().0.as_deref(), Some("<special>"));
    }

    #[test]
    fn prompt_tokens_provide_context_without_re_emission() {
        let backend = Utf8Backend;
        let prompt = &[b'H' as u32, b'i' as u32];
        let mut decoder = backend.create_decode_stream(prompt, false, 0);

        // First generated token should not re-emit "Hi".
        let (chunk, _) = decoder.push_token(b'!' as u32).unwrap();
        assert_eq!(chunk.as_deref(), Some("!"));
    }

    #[test]
    fn chunks_concatenate_to_full_text() {
        let backend = Utf8Backend;
        let mut decoder = backend.create_decode_stream(&[], false, 0);

        let input = b"Hello, world!";
        let mut full = String::new();
        for &byte in input {
            if let (Some(chunk), _) = decoder.push_token(byte as u32).unwrap() {
                full.push_str(&chunk);
            }
        }
        if let Some(chunk) = decoder.flush().unwrap() {
            full.push_str(&chunk);
        }
        assert_eq!(full, "Hello, world!");
    }
}
