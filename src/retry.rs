// SPDX-License-Identifier: ISC
// SPDX-FileCopyrightText: 2026 Murilo Ijanc' <murilo@ijanc.org>

//! The [`RetryProvider`] decorator: transparent, transient-failure retries.
//!
//! A [`RetryProvider`] wraps any inner [`Provider`] and retries a failed
//! [`complete`](Provider::complete), or the establishment of a
//! [`complete_stream`](Provider::complete_stream), when the failure is
//! transient and happened before any stream content was yielded. Each retry
//! waits out a jittered exponential backoff, honoring a server-requested
//! `Retry-After` over its own schedule, and every wait is an await point so a
//! cancelled call stops waiting at once.
//!
//! Waiting is factored behind the [`Clock`] seam so tests drive time without
//! sleeping and callers can supply their own timer; the default
//! [`SystemClock`] backs each wait with a dedicated timer thread, keeping the
//! decorator runtime-agnostic.

use crate::error::{Error, ErrorKind};
use crate::message::AssistantMessage;
use crate::provider::Provider;
use crate::request::{CompletionOptions, Context};
use crate::stream::{StreamEvent, StreamEvents};
use async_trait::async_trait;
use futures_util::StreamExt;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};
use std::time::Duration;

/// The waiting seam a [`RetryProvider`] backs off against.
///
/// Factoring the wait out lets a test record requested delays without sleeping,
/// and lets a caller on a specific runtime supply its own timer. The wait is an
/// `async fn` so it is an await point: dropping (cancelling) the enclosing call
/// stops the wait immediately.
#[async_trait]
pub trait Clock: Send + Sync {
    /// Wait for `duration`.
    async fn sleep(&self, duration: Duration);
}

/// The default [`Clock`], backing each wait with a dedicated timer thread.
///
/// A timer thread keeps the crate runtime-agnostic: it needs no async
/// executor's timer, so the same [`RetryProvider`] runs unchanged on any
/// runtime. Waits fire only on a retryable failure, so the per-wait thread is
/// off the hot path.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

#[async_trait]
impl Clock for SystemClock {
    async fn sleep(&self, duration: Duration) {
        if duration.is_zero() {
            return;
        }
        ThreadTimer::new(duration).await;
    }
}

/// Shared handoff between a timer thread and the task awaiting it.
#[derive(Debug, Default)]
struct TimerShared {
    /// Set once the timer thread's sleep has elapsed.
    fired: AtomicBool,
    /// The waker of the task parked on this timer, if it has polled.
    waker: Mutex<Option<Waker>>,
}

/// A one-shot async sleep that resolves when its timer thread fires.
///
/// On first poll it spawns a thread that sleeps the requested duration, then
/// sets [`TimerShared::fired`] and wakes the parked task. Dropping the future
/// leaves the thread to finish harmlessly, so a cancelled wait returns at once.
struct ThreadTimer {
    shared: Arc<TimerShared>,
    duration: Duration,
    started: bool,
}

impl ThreadTimer {
    fn new(duration: Duration) -> Self {
        Self {
            shared: Arc::new(TimerShared::default()),
            duration,
            started: false,
        }
    }
}

impl Future for ThreadTimer {
    type Output = ();

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<()> {
        if self.shared.fired.load(Ordering::Acquire) {
            return Poll::Ready(());
        }
        // Register the current waker before (re)checking `fired`, so a fire
        // racing this poll still wakes us.
        *self.shared.waker.lock().unwrap() = Some(cx.waker().clone());

        if !self.started {
            self.started = true;
            let shared = Arc::clone(&self.shared);
            let duration = self.duration;
            std::thread::spawn(move || {
                std::thread::sleep(duration);
                shared.fired.store(true, Ordering::Release);
                if let Some(waker) = shared.waker.lock().unwrap().take() {
                    waker.wake();
                }
            });
        }

        // The thread may have fired between the load above and registering the
        // waker; re-check so the wake is never missed.
        if self.shared.fired.load(Ordering::Acquire) {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    }
}

/// How a [`RetryProvider`] spaces and bounds its retries.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Retries attempted after the first try; total attempts is this plus one.
    pub max_retries: u32,
    /// Backoff before the first retry, doubled on each subsequent retry.
    pub base_delay: Duration,
    /// Ceiling on any single wait. A server-requested delay above this fails
    /// fast instead of parking for it.
    pub max_delay: Duration,
    /// Fraction of the computed backoff to remove at random as jitter, in
    /// `0.0..=1.0`. Zero makes waits deterministic.
    pub jitter: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(60),
            jitter: 0.25,
        }
    }
}

impl RetryPolicy {
    /// The base backoff for `retry` (0-based: the first retry is `0`), doubled
    /// per step and capped at [`max_delay`](Self::max_delay), before jitter.
    fn backoff(&self, retry: u32) -> Duration {
        let factor = 2u32.saturating_pow(retry);
        self.base_delay.saturating_mul(factor).min(self.max_delay)
    }

    /// Remove up to [`jitter`](Self::jitter) of `base` at random, given a draw
    /// in `0.0..1.0`. A zero jitter (or draw) leaves `base` untouched; a jitter
    /// outside `0.0..=1.0` is clamped so a stray value cannot invert or inflate
    /// the wait.
    fn jittered(&self, base: Duration, draw: f64) -> Duration {
        let jitter = self.jitter.clamp(0.0, 1.0);
        if jitter <= 0.0 {
            return base;
        }
        base.saturating_sub(base.mul_f64(jitter * draw))
    }
}

/// Whether an error is worth another attempt.
///
/// A retryable HTTP status — request timeout (408), conflict (409), rate limit
/// (429), or any 5xx — or a transport-level failure (a connection drop,
/// timeout, or DNS/TLS error before a response arrived) is retryable; a
/// malformed request, an auth failure, or a decode error is not.
fn is_retryable(error: &Error) -> bool {
    if let Some(status) = error.status() {
        return matches!(status, 408 | 409 | 429) || status >= 500;
    }
    matches!(error.kind(), ErrorKind::Transport)
}

/// A non-cryptographic draw in `0.0..1.0` for backoff jitter.
///
/// Jitter only needs to de-correlate concurrent waiters, not resist an
/// adversary, so a process-wide xorshift generator is enough and keeps the
/// crate free of a randomness dependency.
fn jitter_draw() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    static STATE: AtomicU64 = AtomicU64::new(0);
    let mut x = STATE.load(Ordering::Relaxed);
    if x == 0 {
        // Seed lazily from the wall clock the first time (or, harmlessly, when
        // racing callers each reseed); `| 1` keeps the state non-zero.
        x = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0x9E37_79B9_7F4A_7C15, |d| d.as_nanos() as u64)
            | 1;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    STATE.store(x, Ordering::Relaxed);
    // The top 53 bits map exactly onto an f64 in `[0, 1)`.
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// A [`Provider`] decorator that retries transient failures transparently.
///
/// It wraps any inner Provider and retries a failed
/// [`complete`](Provider::complete), or the establishment of a
/// [`complete_stream`](Provider::complete_stream), when the failure is
/// transient — a retryable status (408/409/429/5xx) or a transport error — and
/// happened before any stream content was yielded. Once a stream has yielded
/// its first event, a later failure flows through untouched: mid-stream errors
/// are never retried. Each retry waits out a jittered exponential backoff,
/// honoring a server-requested `Retry-After` over its own schedule; a
/// server-requested delay beyond [`RetryPolicy::max_delay`] fails fast rather
/// than parking for it. Every wait is an await point, so a cancelled call stops
/// waiting at once.
///
/// `Retry-After` is honored whenever the failing error carries one (the
/// buffered [`complete`](Provider::complete) path attaches it); a streaming
/// failure raised before its response headers surface simply falls back to the
/// backoff schedule.
///
/// The decorator is single-provider: it retries the same inner Provider with
/// the same Credential.
//
// TODO(failover): this is the seam for multi-provider failover and API-key
// rotation. A future decorator wraps several inner Providers (or several
// Credentials) and, when the retries here are exhausted or the failure is not
// retryable, advances to the next one before giving up. Keeping it beside this
// type lets the retry loop and the failover loop compose rather than entangle.
pub struct RetryProvider<P> {
    inner: P,
    policy: RetryPolicy,
    clock: Arc<dyn Clock>,
}

impl<P: Provider> RetryProvider<P> {
    /// Wrap `inner`, retrying per `policy` and waiting on the default
    /// [`SystemClock`].
    pub fn new(inner: P, policy: RetryPolicy) -> Self {
        Self::with_clock(inner, policy, Arc::new(SystemClock))
    }

    /// Wrap `inner` with an explicit [`Clock`], so a caller — or a test — drives
    /// the waits themselves.
    pub fn with_clock(
        inner: P,
        policy: RetryPolicy,
        clock: Arc<dyn Clock>,
    ) -> Self {
        Self {
            inner,
            policy,
            clock,
        }
    }

    /// How long to wait before the next attempt, or `None` to stop.
    ///
    /// Returns `None` when the error is not retryable, no retries remain, or a
    /// server-requested delay exceeds [`RetryPolicy::max_delay`] (fail fast).
    fn wait_for(&self, error: &Error, retry: u32) -> Option<Duration> {
        if retry >= self.policy.max_retries || !is_retryable(error) {
            return None;
        }
        if let Some(requested) = error.retry_after() {
            // A server asking for longer than we will ever wait gets no wait:
            // surface the error now instead of parking for it.
            return (requested <= self.policy.max_delay).then_some(requested);
        }
        Some(
            self.policy
                .jittered(self.policy.backoff(retry), jitter_draw()),
        )
    }

    /// Weigh a failed attempt: wait out the backoff and bump `retry` to try
    /// again, or give the error back to stop.
    ///
    /// Both the `complete` and `complete_stream` loops share this arm, so the
    /// retry accounting and the (cancellable) wait live in one place.
    async fn after_failure(
        &self,
        error: Error,
        retry: &mut u32,
    ) -> Result<(), Error> {
        match self.wait_for(&error, *retry) {
            Some(delay) => {
                self.clock.sleep(delay).await;
                *retry += 1;
                Ok(())
            }
            None => Err(error),
        }
    }

    /// Establish a stream and peek its first item, so a pre-content failure can
    /// be retried while a started stream is handed back intact.
    ///
    /// A failure to open the stream, or an error as its very first item, is
    /// returned as `Err` for the retry loop to weigh. Once a real event has
    /// arrived it is replayed ahead of the rest of the stream, and no further
    /// retry can happen.
    async fn establish_stream(
        &self,
        ctx: &Context,
        opts: &CompletionOptions,
    ) -> Result<StreamEvents, Error> {
        let mut stream = self.inner.complete_stream(ctx, opts).await?;
        match stream.next().await {
            Some(Ok(event)) => {
                let head =
                    futures_util::stream::once(
                        async move { Ok::<_, Error>(event) },
                    );
                Ok(Box::pin(head.chain(stream)))
            }
            Some(Err(error)) => Err(error),
            None => Ok(Box::pin(futures_util::stream::empty::<
                Result<StreamEvent, Error>,
            >())),
        }
    }
}

#[async_trait]
impl<P: Provider> Provider for RetryProvider<P> {
    async fn complete(
        &self,
        ctx: &Context,
        opts: &CompletionOptions,
    ) -> Result<AssistantMessage, Error> {
        let mut retry = 0;
        loop {
            match self.inner.complete(ctx, opts).await {
                Ok(response) => return Ok(response),
                Err(error) => self.after_failure(error, &mut retry).await?,
            }
        }
    }

    async fn complete_stream(
        &self,
        ctx: &Context,
        opts: &CompletionOptions,
    ) -> Result<StreamEvents, Error> {
        let mut retry = 0;
        loop {
            match self.establish_stream(ctx, opts).await {
                Ok(stream) => return Ok(stream),
                Err(error) => self.after_failure(error, &mut retry).await?,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
    use crate::response::{FinishReason, Usage};
    use std::collections::VecDeque;
    use std::sync::atomic::AtomicUsize;

    /// A fast policy for tests: no jitter, tiny delays, so recorded waits are
    /// exact.
    fn test_policy(max_retries: u32) -> RetryPolicy {
        RetryPolicy {
            max_retries,
            base_delay: Duration::from_millis(10),
            max_delay: Duration::from_secs(1),
            jitter: 0.0,
        }
    }

    /// A [`Clock`] that records requested waits without ever sleeping.
    #[derive(Debug, Default)]
    struct RecordingClock {
        waits: Mutex<Vec<Duration>>,
    }

    impl RecordingClock {
        fn waits(&self) -> Vec<Duration> {
            self.waits.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Clock for RecordingClock {
        async fn sleep(&self, duration: Duration) {
            self.waits.lock().unwrap().push(duration);
        }
    }

    /// A [`Clock`] whose wait never resolves, to exercise cancellation.
    struct BlockingClock;

    #[async_trait]
    impl Clock for BlockingClock {
        async fn sleep(&self, _duration: Duration) {
            std::future::pending::<()>().await;
        }
    }

    /// One queued streaming outcome for the test double.
    enum StreamOutcome {
        /// `complete_stream` fails before yielding a stream.
        Establish(Error),
        /// A stream that yields these items in order.
        Events(Vec<Result<StreamEvent, Error>>),
    }

    /// A Provider that replays a queued sequence of outcomes, one per call, so a
    /// test drives the retry loop deterministically.
    #[derive(Default)]
    struct SequencedProvider {
        completions: Mutex<VecDeque<Result<AssistantMessage, Error>>>,
        streams: Mutex<VecDeque<StreamOutcome>>,
        calls: AtomicUsize,
    }

    impl SequencedProvider {
        fn with_completions(
            outcomes: Vec<Result<AssistantMessage, Error>>,
        ) -> Self {
            Self {
                completions: Mutex::new(outcomes.into()),
                ..Self::default()
            }
        }

        fn with_streams(outcomes: Vec<StreamOutcome>) -> Self {
            Self {
                streams: Mutex::new(outcomes.into()),
                ..Self::default()
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Provider for SequencedProvider {
        async fn complete(
            &self,
            _ctx: &Context,
            _opts: &CompletionOptions,
        ) -> Result<AssistantMessage, Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.completions
                .lock()
                .unwrap()
                .pop_front()
                .expect("SequencedProvider ran out of queued completions")
        }

        async fn complete_stream(
            &self,
            _ctx: &Context,
            _opts: &CompletionOptions,
        ) -> Result<StreamEvents, Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match self
                .streams
                .lock()
                .unwrap()
                .pop_front()
                .expect("SequencedProvider ran out of queued streams")
            {
                StreamOutcome::Establish(error) => Err(error),
                StreamOutcome::Events(items) => {
                    Ok(Box::pin(futures_util::stream::iter(items)))
                }
            }
        }
    }

    fn ok_response(text: &str) -> AssistantMessage {
        AssistantMessage::text(text)
    }

    /// The context every test drives the retry loop with.
    fn ctx() -> Context {
        Context::new(vec![Message::user("hi")])
    }

    /// The options every test drives the retry loop with.
    fn opts() -> CompletionOptions {
        CompletionOptions::default()
    }

    #[tokio::test]
    async fn retries_a_rate_limit_then_succeeds() {
        let inner = Arc::new(SequencedProvider::with_completions(vec![
            Err(Error::from_status(429, "slow down")),
            Err(Error::from_status(503, "overloaded")),
            Ok(ok_response("done")),
        ]));
        let clock = Arc::new(RecordingClock::default());
        let retry = RetryProvider::with_clock(
            inner.clone(),
            test_policy(3),
            clock.clone(),
        );

        let response = retry.complete(&ctx(), &opts()).await.unwrap();

        assert_eq!(response.text_content(), "done");
        // Three calls: the two failures and the success.
        assert_eq!(inner.calls(), 3);
        // Backoff doubled between the two retries: 10ms then 20ms.
        assert_eq!(
            clock.waits(),
            vec![Duration::from_millis(10), Duration::from_millis(20)]
        );
    }

    #[tokio::test]
    async fn honors_retry_after_over_computed_backoff() {
        let inner = Arc::new(SequencedProvider::with_completions(vec![
            Err(Error::from_status(429, "slow down")
                .with_retry_after(Duration::from_millis(750))),
            Ok(ok_response("done")),
        ]));
        let clock = Arc::new(RecordingClock::default());
        let retry = RetryProvider::with_clock(
            inner.clone(),
            test_policy(3),
            clock.clone(),
        );

        retry.complete(&ctx(), &opts()).await.unwrap();

        // The server's 750ms wins over the 10ms backoff.
        assert_eq!(clock.waits(), vec![Duration::from_millis(750)]);
    }

    #[tokio::test]
    async fn retry_after_above_the_cap_fails_fast() {
        let inner = Arc::new(SequencedProvider::with_completions(vec![
            // Asks for longer than the 1s cap.
            Err(Error::from_status(429, "slow down")
                .with_retry_after(Duration::from_secs(5))),
            Ok(ok_response("unreached")),
        ]));
        let clock = Arc::new(RecordingClock::default());
        let retry = RetryProvider::with_clock(
            inner.clone(),
            test_policy(3),
            clock.clone(),
        );

        let error = retry.complete(&ctx(), &opts()).await.unwrap_err();

        assert_eq!(error.kind(), ErrorKind::RateLimited);
        // Failed fast: the first attempt only, and no wait.
        assert_eq!(inner.calls(), 1);
        assert!(clock.waits().is_empty());
    }

    #[tokio::test]
    async fn stops_after_exhausting_the_retries() {
        let inner = Arc::new(SequencedProvider::with_completions(vec![
            Err(Error::from_status(503, "a")),
            Err(Error::from_status(503, "b")),
            Err(Error::from_status(503, "c")),
            Err(Error::from_status(503, "d")),
        ]));
        let clock = Arc::new(RecordingClock::default());
        let retry = RetryProvider::with_clock(
            inner.clone(),
            test_policy(2),
            clock.clone(),
        );

        let error = retry.complete(&ctx(), &opts()).await.unwrap_err();

        assert_eq!(error.kind(), ErrorKind::ServerError);
        // One initial attempt plus two retries.
        assert_eq!(inner.calls(), 3);
        assert_eq!(clock.waits().len(), 2);
    }

    #[tokio::test]
    async fn does_not_retry_a_non_retryable_error() {
        let inner = Arc::new(SequencedProvider::with_completions(vec![
            Err(Error::from_status(400, "bad request")),
            Ok(ok_response("unreached")),
        ]));
        let clock = Arc::new(RecordingClock::default());
        let retry = RetryProvider::with_clock(
            inner.clone(),
            test_policy(3),
            clock.clone(),
        );

        let error = retry.complete(&ctx(), &opts()).await.unwrap_err();

        assert_eq!(error.kind(), ErrorKind::InvalidRequest);
        assert_eq!(inner.calls(), 1);
        assert!(clock.waits().is_empty());
    }

    #[tokio::test]
    async fn a_transport_error_is_retried() {
        let inner = Arc::new(SequencedProvider::with_completions(vec![
            Err(Error::new(ErrorKind::Transport, "connection reset")),
            Ok(ok_response("done")),
        ]));
        let clock = Arc::new(RecordingClock::default());
        let retry = RetryProvider::with_clock(
            inner.clone(),
            test_policy(3),
            clock.clone(),
        );

        let response = retry.complete(&ctx(), &opts()).await.unwrap();
        assert_eq!(response.text_content(), "done");
        assert_eq!(inner.calls(), 2);
    }

    #[tokio::test]
    async fn a_cancelled_call_stops_waiting_promptly() {
        let inner = Arc::new(SequencedProvider::with_completions(vec![
            Err(Error::from_status(429, "slow down")),
            Ok(ok_response("unreached")),
        ]));
        let retry = RetryProvider::with_clock(
            inner.clone(),
            test_policy(3),
            Arc::new(BlockingClock),
        );

        let (ctx, opts) = (ctx(), opts());
        let fut = retry.complete(&ctx, &opts);
        tokio::pin!(fut);
        // The first attempt fails and the call parks in the (forever) wait;
        // racing it against an immediately-ready future drops it mid-wait.
        let cancelled = tokio::select! {
            biased;
            _ = &mut fut => false,
            () = std::future::ready(()) => true,
        };

        assert!(cancelled, "the wait should still be pending");
        // Cancelling stopped the wait before the second attempt could run.
        assert_eq!(inner.calls(), 1);
    }

    #[tokio::test]
    async fn retries_stream_establishment_then_succeeds() {
        let inner = Arc::new(SequencedProvider::with_streams(vec![
            StreamOutcome::Establish(Error::from_status(503, "overloaded")),
            StreamOutcome::Events(vec![
                Ok(StreamEvent::MessageStart),
                Ok(StreamEvent::TextDelta {
                    index: 0,
                    text: "hi".to_owned(),
                }),
            ]),
        ]));
        let clock = Arc::new(RecordingClock::default());
        let retry = RetryProvider::with_clock(
            inner.clone(),
            test_policy(3),
            clock.clone(),
        );

        let events: Vec<_> = retry
            .complete_stream(&ctx(), &opts())
            .await
            .unwrap()
            .map(Result::unwrap)
            .collect()
            .await;

        assert_eq!(events.first(), Some(&StreamEvent::MessageStart));
        assert_eq!(events.len(), 2);
        assert_eq!(inner.calls(), 2);
        assert_eq!(clock.waits(), vec![Duration::from_millis(10)]);
    }

    #[tokio::test]
    async fn retries_when_the_first_stream_item_is_a_failure() {
        let inner = Arc::new(SequencedProvider::with_streams(vec![
            // The stream opens, then errors before any event.
            StreamOutcome::Events(vec![Err(Error::from_status(429, "slow"))]),
            StreamOutcome::Events(vec![
                Ok(StreamEvent::MessageStart),
                Ok(StreamEvent::Done {
                    finish_reason: FinishReason::Stop,
                    usage: Usage::default(),
                }),
            ]),
        ]));
        let clock = Arc::new(RecordingClock::default());
        let retry = RetryProvider::with_clock(
            inner.clone(),
            test_policy(3),
            clock.clone(),
        );

        let events: Vec<_> = retry
            .complete_stream(&ctx(), &opts())
            .await
            .unwrap()
            .map(Result::unwrap)
            .collect()
            .await;

        assert_eq!(events.first(), Some(&StreamEvent::MessageStart));
        assert_eq!(inner.calls(), 2);
        assert_eq!(clock.waits().len(), 1);
    }

    #[tokio::test]
    async fn a_mid_stream_failure_is_not_retried() {
        let inner = Arc::new(SequencedProvider::with_streams(vec![
            // A real event, then a retryable-looking failure mid-stream.
            StreamOutcome::Events(vec![
                Ok(StreamEvent::MessageStart),
                Err(Error::from_status(503, "dropped mid-stream")),
            ]),
            StreamOutcome::Events(vec![Ok(StreamEvent::MessageStart)]),
        ]));
        let clock = Arc::new(RecordingClock::default());
        let retry = RetryProvider::with_clock(
            inner.clone(),
            test_policy(3),
            clock.clone(),
        );

        let items: Vec<_> = retry
            .complete_stream(&ctx(), &opts())
            .await
            .unwrap()
            .collect()
            .await;

        // The started stream is handed back intact: the event, then its error.
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].as_ref().unwrap(), &StreamEvent::MessageStart);
        assert!(items[1].is_err());
        // No second attempt, no wait.
        assert_eq!(inner.calls(), 1);
        assert!(clock.waits().is_empty());
    }

    #[test]
    fn backoff_doubles_and_caps() {
        let policy = RetryPolicy {
            max_retries: 10,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(5),
            jitter: 0.0,
        };
        assert_eq!(policy.backoff(0), Duration::from_secs(1));
        assert_eq!(policy.backoff(1), Duration::from_secs(2));
        assert_eq!(policy.backoff(2), Duration::from_secs(4));
        // Capped at max_delay from here on.
        assert_eq!(policy.backoff(3), Duration::from_secs(5));
        assert_eq!(policy.backoff(30), Duration::from_secs(5));
    }

    #[test]
    fn jitter_removes_at_most_its_fraction() {
        let policy = RetryPolicy {
            jitter: 0.5,
            ..RetryPolicy::default()
        };
        let base = Duration::from_secs(10);
        // A full draw removes the whole 50% fraction; a zero draw removes none.
        assert_eq!(policy.jittered(base, 1.0), Duration::from_secs(5));
        assert_eq!(policy.jittered(base, 0.0), base);
        // Every real draw stays within [base - jitter*base, base].
        for _ in 0..1000 {
            let d = policy.jittered(base, jitter_draw());
            assert!(d <= base && d >= Duration::from_secs(5));
        }
    }

    #[test]
    fn out_of_range_jitter_is_clamped() {
        let base = Duration::from_secs(10);
        // Above 1.0 clamps to a full fraction: a full draw removes all of base,
        // never more.
        let high = RetryPolicy {
            jitter: 2.0,
            ..RetryPolicy::default()
        };
        assert_eq!(high.jittered(base, 1.0), Duration::ZERO);
        // Below 0.0 clamps to no jitter.
        let low = RetryPolicy {
            jitter: -1.0,
            ..RetryPolicy::default()
        };
        assert_eq!(low.jittered(base, 1.0), base);
    }

    #[test]
    fn is_retryable_classifies_by_status_and_kind() {
        for status in [408, 409, 429, 500, 503, 529] {
            assert!(is_retryable(&Error::from_status(status, "")));
        }
        for status in [400, 401, 403, 404, 422] {
            assert!(!is_retryable(&Error::from_status(status, "")));
        }
        assert!(is_retryable(&Error::new(ErrorKind::Transport, "")));
        assert!(!is_retryable(&Error::new(ErrorKind::Decode, "")));
    }

    #[tokio::test]
    async fn system_clock_wait_resolves() {
        // The real timer path fires and returns.
        SystemClock.sleep(Duration::from_millis(5)).await;
        // A zero wait short-circuits without spawning a thread.
        SystemClock.sleep(Duration::ZERO).await;
    }
}
