// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! A pure Server-Sent Events decoder over byte chunks.
//!
//! [`SseDecoder`] is fed raw byte chunks as they arrive off the transport and
//! yields whole [`SseEvent`]s. It holds no async or network concerns: bytes go
//! in, events come out, and an event split across chunk boundaries is buffered
//! until it is complete. This is the seam between the byte stream and the
//! Provider-specific normalization that turns events into a stream vocabulary.

/// One decoded Server-Sent Event: an optional `event:` type and the joined
/// `data:` payload.
///
/// Per the SSE grammar, multiple `data:` fields in one event are joined with a
/// newline, and an event with no `event:` field carries `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field, if the event carried one.
    pub event: Option<String>,
    /// The concatenated `data:` payload, newline-joined across `data:` lines.
    pub data: String,
}

/// Incrementally decodes SSE byte chunks into whole [`SseEvent`]s.
///
/// Feed chunks with [`SseDecoder::push`]; each call returns every event that
/// became complete within it. A trailing partial line or a not-yet-terminated
/// event is retained across calls, so events that straddle chunk boundaries are
/// reassembled transparently.
#[derive(Debug, Default)]
pub struct SseDecoder {
    /// Bytes received but not yet forming a complete line.
    buffer: Vec<u8>,
    /// The `event:` field seen for the event under construction.
    event: Option<String>,
    /// The `data:` lines accumulated for the event under construction.
    data: Vec<String>,
    /// Whether any field line has been seen since the last dispatch, so a bare
    /// blank line between events does not emit an empty event.
    started: bool,
}

impl SseDecoder {
    /// A fresh decoder with an empty buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed a chunk of bytes, returning every event completed by it.
    ///
    /// Bytes that do not yet form a full line, or a full-but-unterminated
    /// event, are held for the next call.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        self.buffer.extend_from_slice(chunk);

        let mut events = Vec::new();
        // Consume every complete line (terminated by `\n`) in the buffer.
        while let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..=pos).collect();
            line.pop(); // drop the '\n'
            if line.last() == Some(&b'\r') {
                line.pop(); // drop a preceding '\r' (CRLF)
            }
            if let Some(event) = self.consume_line(&line) {
                events.push(event);
            }
        }
        events
    }

    /// Flush an event that was fully received but not blank-line terminated
    /// before the stream ended.
    ///
    /// Well-formed SSE ends each event with a blank line, but a stream may close
    /// straight after the last event's fields; this surfaces that final event.
    pub fn finish(&mut self) -> Option<SseEvent> {
        // A trailing partial line (no '\n') still counts as a field line.
        if !self.buffer.is_empty() {
            let mut line = std::mem::take(&mut self.buffer);
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let _ = self.consume_line(&line);
        }
        self.dispatch()
    }

    /// Process one decoded line, returning an event if the line was blank and
    /// closed an in-progress event.
    fn consume_line(&mut self, line: &[u8]) -> Option<SseEvent> {
        if line.is_empty() {
            return self.dispatch();
        }
        // Comment lines start with ':' and are ignored.
        if line.first() == Some(&b':') {
            return None;
        }

        let text = String::from_utf8_lossy(line);
        let (field, value) = match text.split_once(':') {
            // A single space after the colon is part of the syntax, not data.
            Some((field, value)) => {
                (field, value.strip_prefix(' ').unwrap_or(value))
            }
            // A line with no colon is a field with an empty value.
            None => (text.as_ref(), ""),
        };

        match field {
            "event" => {
                self.event = Some(value.to_owned());
                self.started = true;
            }
            "data" => {
                self.data.push(value.to_owned());
                self.started = true;
            }
            // `id` and `retry` carry no meaning for this decoder; ignore other
            // fields but still count the event as started.
            _ => self.started = true,
        }
        None
    }

    /// Emit the in-progress event and reset for the next one.
    fn dispatch(&mut self) -> Option<SseEvent> {
        if !self.started {
            return None;
        }
        let event = self.event.take();
        let data = self.data.join("\n");
        self.data.clear();
        self.started = false;
        Some(SseEvent { event, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain(decoder: &mut SseDecoder, chunks: &[&[u8]]) -> Vec<SseEvent> {
        let mut out = Vec::new();
        for chunk in chunks {
            out.extend(decoder.push(chunk));
        }
        out.extend(decoder.finish());
        out
    }

    #[test]
    fn decodes_a_single_event_with_type_and_data() {
        let mut decoder = SseDecoder::new();
        let events = drain(
            &mut decoder,
            &[b"event: message_start\ndata: {\"a\":1}\n\n"],
        );
        assert_eq!(
            events,
            vec![SseEvent {
                event: Some("message_start".to_owned()),
                data: "{\"a\":1}".to_owned(),
            }]
        );
    }

    #[test]
    fn joins_multiple_data_lines_with_newlines() {
        let mut decoder = SseDecoder::new();
        let events =
            drain(&mut decoder, &[b"data: line one\ndata: line two\n\n"]);
        assert_eq!(events[0].data, "line one\nline two");
        assert_eq!(events[0].event, None);
    }

    #[test]
    fn reassembles_an_event_split_across_chunk_boundaries() {
        let mut decoder = SseDecoder::new();
        // The split falls inside the `event:` field name and the `data:` value.
        let events = drain(
            &mut decoder,
            &[b"eve", b"nt: text\nda", b"ta: he", b"llo\n", b"\n"],
        );
        assert_eq!(
            events,
            vec![SseEvent {
                event: Some("text".to_owned()),
                data: "hello".to_owned(),
            }]
        );
    }

    #[test]
    fn splits_multiple_events_within_one_chunk() {
        let mut decoder = SseDecoder::new();
        let events = drain(
            &mut decoder,
            &[b"event: a\ndata: 1\n\nevent: b\ndata: 2\n\n"],
        );
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event.as_deref(), Some("a"));
        assert_eq!(events[0].data, "1");
        assert_eq!(events[1].event.as_deref(), Some("b"));
        assert_eq!(events[1].data, "2");
    }

    #[test]
    fn handles_crlf_line_endings() {
        let mut decoder = SseDecoder::new();
        let events = drain(&mut decoder, &[b"event: ping\r\ndata: {}\r\n\r\n"]);
        assert_eq!(events[0].event.as_deref(), Some("ping"));
        assert_eq!(events[0].data, "{}");
    }

    #[test]
    fn ignores_comment_lines() {
        let mut decoder = SseDecoder::new();
        let events =
            drain(&mut decoder, &[b": this is a heartbeat\ndata: x\n\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "x");
    }

    #[test]
    fn flushes_a_final_event_without_a_trailing_blank_line() {
        let mut decoder = SseDecoder::new();
        // No terminating blank line before end of stream.
        let events = drain(&mut decoder, &[b"event: message_stop\ndata: {}\n"]);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("message_stop"));
    }

    #[test]
    fn a_lone_blank_line_emits_nothing() {
        let mut decoder = SseDecoder::new();
        let events = drain(&mut decoder, &[b"\n\n"]);
        assert!(events.is_empty());
    }

    #[test]
    fn a_field_with_no_space_after_colon_keeps_its_value() {
        let mut decoder = SseDecoder::new();
        let events = drain(&mut decoder, &[b"data:tight\n\n"]);
        assert_eq!(events[0].data, "tight");
    }
}
