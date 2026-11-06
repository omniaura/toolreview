// Package jev calls TypeSafe's Jev decision model through OpenRouter's
// decisions endpoint.
//
// Jev is not a chat model. A request carries a state (any JSON) and named
// multiple-choice questions about it; the response is one typed answer per
// question with a calibrated probability per label. It writes no text, which
// makes it fast (about 200ms) and cheap enough to review a whole tool catalog.
package jev

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/omniaura/toolreview"
)

// DefaultModel is pinned to a version rather than the ~typesafe/jev-latest
// alias: thresholds are tuned against one model's calibration.
const DefaultModel = "typesafe/jev-1.13"

// DefaultURL is OpenRouter's decisions endpoint.
const DefaultURL = "https://openrouter.ai/api/alpha/decisions"

const (
	defaultTimeout        = 10 * time.Second
	maxResponseBytes      = 1 << 20
	maxUpstreamErrorBytes = 512
	maxQuestions          = 32
)

// ErrNotConfigured means the client has no API key. Retrying cannot help.
var ErrNotConfigured = errors.New("jev: API key is not configured")

// ErrNoAnswer means the response did not answer the question asked.
var ErrNoAnswer = errors.New("jev: no answer for question")

// HTTPError is an upstream rejection. Body is bounded diagnostic text.
type HTTPError struct {
	Body       string
	StatusCode int
}

func (e *HTTPError) Error() string {
	return "jev: upstream status " + strconv.Itoa(e.StatusCode) + ": " + e.Body
}

// Retryable reports whether the request may succeed if sent again.
func (e *HTTPError) Retryable() bool {
	return e.StatusCode == http.StatusTooManyRequests || e.StatusCode >= 500
}

// Client talks to the decisions endpoint. Build one with New.
type Client struct {
	http    *http.Client
	apiKey  func() string
	headers http.Header
	url     string
	model   string
}

// Option configures a Client.
type Option func(*Client)

// WithHTTPClient replaces the default HTTP client.
func WithHTTPClient(h *http.Client) Option {
	return func(c *Client) {
		if h != nil {
			c.http = h
		}
	}
}

// WithURL overrides the decisions endpoint.
func WithURL(url string) Option {
	return func(c *Client) {
		if url = strings.TrimSpace(url); url != "" {
			c.url = url
		}
	}
}

// WithModel overrides the decision model.
func WithModel(model string) Option {
	return func(c *Client) {
		if model = strings.TrimSpace(model); model != "" {
			c.model = model
		}
	}
}

// WithHeader adds a header to every request, such as OpenRouter's
// HTTP-Referer and X-Title attribution.
func WithHeader(key, value string) Option {
	return func(c *Client) { c.headers.Add(key, value) }
}

// New builds a client. apiKey is read on every call, so a rotated secret is
// picked up without a restart.
func New(apiKey func() string, opts ...Option) *Client {
	c := &Client{
		http:    &http.Client{Timeout: defaultTimeout},
		apiKey:  apiKey,
		headers: http.Header{},
		url:     DefaultURL,
		model:   DefaultModel,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Model is the model the client asks.
func (c *Client) Model() string { return c.model }

// Question is one multiple-choice question about a state.
type Question struct {
	// Criteria maps each label to when it applies. Answers are one of these.
	Criteria map[string]string
	// Instructions is the question.
	Instructions string
}

// Answer is Jev's answer to a Question.
type Answer struct {
	Probabilities map[string]float64
	Label         string
	Confidence    float64
}

// Usage is the accounting of one request.
type Usage struct {
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
	Cost         float64 `json:"cost"`
}

// Batch is the answers to one request and its usage.
type Batch struct {
	Answers map[string]Answer
	Usage   Usage
}

type wireRequest struct {
	State     any                     `json:"state"`
	Questions map[string]wireQuestion `json:"questions"`
	Model     string                  `json:"model"`
}

type wireQuestion struct {
	Criteria     map[string]string `json:"criteria"`
	Type         string            `json:"type"`
	Instructions string            `json:"instructions"`
}

type wireResponse struct {
	Answers map[string]wireAnswer `json:"answers"`
	Usage   Usage                 `json:"usage"`
}

type wireAnswer struct {
	Probabilities map[string]float64 `json:"probabilities"`
	Choice        string             `json:"choice"`
	Confidence    float64            `json:"confidence"`
}

// Ask sends named questions about one state in a single request. An answer
// whose label was not declared is dropped, so a caller never acts on a label
// it did not offer.
func (c *Client) Ask(ctx context.Context, state any, questions map[string]Question) (Batch, error) {
	if c == nil || c.apiKey == nil {
		return Batch{}, ErrNotConfigured
	}
	key := strings.TrimSpace(c.apiKey())
	if key == "" {
		return Batch{}, ErrNotConfigured
	}
	if len(questions) == 0 || len(questions) > maxQuestions {
		return Batch{}, fmt.Errorf("jev: one to %d questions required", maxQuestions)
	}
	wire := make(map[string]wireQuestion, len(questions))
	for name, q := range questions {
		if name == "" || len(q.Criteria) < 2 {
			return Batch{}, errors.New("jev: a named question needs at least two labels")
		}
		wire[name] = wireQuestion{Type: "choice", Instructions: q.Instructions, Criteria: q.Criteria}
	}
	body, err := json.Marshal(wireRequest{Model: c.model, State: state, Questions: wire})
	if err != nil {
		return Batch{}, fmt.Errorf("jev: encode request: %w", err)
	}
	if _, bounded := ctx.Deadline(); !bounded {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, defaultTimeout)
		defer cancel()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url, bytes.NewReader(body))
	if err != nil {
		return Batch{}, fmt.Errorf("jev: build request: %w", err)
	}
	for k, vs := range c.headers {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	req.Header.Set("Authorization", "Bearer "+key)
	req.Header.Set("Content-Type", "application/json")

	rsp, err := c.http.Do(req)
	if err != nil {
		return Batch{}, fmt.Errorf("jev: request failed: %w", err)
	}
	defer rsp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(rsp.Body, maxResponseBytes+1))
	if err != nil {
		return Batch{}, fmt.Errorf("jev: read response: %w", err)
	}
	if len(raw) > maxResponseBytes {
		return Batch{}, fmt.Errorf("jev: response exceeds %d bytes", maxResponseBytes)
	}
	if rsp.StatusCode != http.StatusOK {
		return Batch{}, &HTTPError{StatusCode: rsp.StatusCode, Body: truncate(raw, maxUpstreamErrorBytes)}
	}
	var decoded wireResponse
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return Batch{}, fmt.Errorf("jev: decode response: %w", err)
	}
	out := Batch{Answers: make(map[string]Answer, len(questions)), Usage: decoded.Usage}
	for name, q := range questions {
		a, ok := decoded.Answers[name]
		if !ok {
			continue
		}
		if _, declared := q.Criteria[a.Choice]; !declared {
			continue
		}
		probs := make(map[string]float64, len(a.Probabilities))
		for label, p := range a.Probabilities {
			if _, declared := q.Criteria[label]; declared {
				probs[label] = p
			}
		}
		out.Answers[name] = Answer{Label: a.Choice, Confidence: a.Confidence, Probabilities: probs}
	}
	return out, nil
}

const singleQuestion = "choice"

// Choose implements toolreview.Chooser.
func (c *Client) Choose(ctx context.Context, state any, instructions string, criteria map[string]string) (toolreview.Choice, error) {
	batch, err := c.Ask(ctx, state, map[string]Question{singleQuestion: {Instructions: instructions, Criteria: criteria}})
	if err != nil {
		return toolreview.Choice{Model: c.model, Cost: batch.Usage.Cost}, err
	}
	a, ok := batch.Answers[singleQuestion]
	if !ok {
		return toolreview.Choice{Model: c.model, Cost: batch.Usage.Cost}, ErrNoAnswer
	}
	return toolreview.Choice{Probabilities: a.Probabilities, Model: c.model, Cost: batch.Usage.Cost}, nil
}

var _ toolreview.Chooser = (*Client)(nil)

func truncate(b []byte, limit int) string {
	if len(b) <= limit {
		return string(b)
	}
	return string(b[:limit]) + "…"
}

// maxRetryPause caps the pause between attempts.
const maxRetryPause = 10 * time.Second

// Retry returns a Chooser that sends a failed call again, up to attempts
// times in all, with a growing pause; a rejection that cannot succeed on retry
// (a 4xx other than 429, or a missing key) is returned at once; an answer
// with no declared label is asked again. The cost of every attempt is
// reported.
func Retry(c *Client, attempts int) toolreview.Chooser {
	if c == nil {
		c = New(nil)
	}
	return retrying{c: c, attempts: max(1, attempts)}
}

type retrying struct {
	c        *Client
	attempts int
}

func (r retrying) Choose(ctx context.Context, state any, instructions string, criteria map[string]string) (toolreview.Choice, error) {
	var last error
	var cost float64
	for attempt := range r.attempts {
		if err := ctx.Err(); err != nil {
			return toolreview.Choice{Model: r.c.model, Cost: cost}, err
		}
		if attempt > 0 {
			t := time.NewTimer(min(time.Duration(attempt*attempt)*time.Second, maxRetryPause))
			select {
			case <-ctx.Done():
				t.Stop()
				return toolreview.Choice{Model: r.c.model, Cost: cost}, ctx.Err()
			case <-t.C:
			}
		}
		choice, err := r.c.Choose(ctx, state, instructions, criteria)
		cost += choice.Cost
		if err == nil {
			choice.Cost = cost
			return choice, nil
		}
		last = err
		var httpErr *HTTPError
		if errors.Is(err, ErrNotConfigured) || (errors.As(err, &httpErr) && !httpErr.Retryable()) {
			break
		}
	}
	return toolreview.Choice{Model: r.c.model, Cost: cost}, last
}
