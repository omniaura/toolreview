package jev

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/omniaura/toolreview"
)

func TestChooseSendsOneChoiceQuestion(t *testing.T) {
	var got wireRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer k" || r.Header.Get("X-Title") != "toolreview" {
			t.Errorf("headers = %v", r.Header)
		}
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &got); err != nil {
			t.Error(err)
		}
		_, _ = w.Write([]byte(`{"answers":{"choice":{"choice":"outbound","confidence":0.7,
			"probabilities":{"read":0.05,"write":0.1,"outbound":0.8,"destructive":0.05,"invented":0.4}}},
			"usage":{"input_tokens":300,"output_tokens":1,"cost":0.0002}}`))
	}))
	defer srv.Close()

	c := New(func() string { return "k" }, WithURL(srv.URL), WithHeader("X-Title", "toolreview"))
	r := toolreview.NewReviewer(c)
	v, err := r.Review(t.Context(), toolreview.Tool{Key: "OUTLOOK_SEND_EMAIL", Name: "OUTLOOK_SEND_EMAIL", Declared: toolreview.Write})
	if err != nil {
		t.Fatal(err)
	}
	if v.Class != toolreview.Outbound || v.Model != DefaultModel || v.Cost != 0.0002 {
		t.Fatalf("verdict = %#v", v)
	}
	if _, ok := v.Probabilities["invented"]; ok {
		t.Fatal("an undeclared label must be dropped")
	}
	q, ok := got.Questions["choice"]
	if got.Model != DefaultModel || !ok || q.Type != "choice" || len(q.Criteria) != 4 || q.Instructions != toolreview.Instructions {
		t.Fatalf("request = %#v", got)
	}
}

func TestUndeclaredChoiceIsNoAnswer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"answers":{"choice":{"choice":"safe","probabilities":{"safe":1}}}}`))
	}))
	defer srv.Close()
	c := New(func() string { return "k" }, WithURL(srv.URL))
	if _, err := c.Choose(t.Context(), map[string]any{}, "q", map[string]string{"a": "", "b": ""}); !errors.Is(err, ErrNoAnswer) {
		t.Fatalf("err = %v", err)
	}
}

func TestHTTPErrorIsTyped(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "slow down", http.StatusTooManyRequests)
	}))
	defer srv.Close()
	c := New(func() string { return "k" }, WithURL(srv.URL))
	_, err := c.Choose(t.Context(), nil, "q", map[string]string{"a": "", "b": ""})
	var httpErr *HTTPError
	if !errors.As(err, &httpErr) || !httpErr.Retryable() {
		t.Fatalf("err = %v", err)
	}
	if _, err := New(func() string { return " " }).Choose(t.Context(), nil, "q", map[string]string{"a": "", "b": ""}); err == nil {
		t.Fatal("a missing key must fail before any request")
	}
}

func TestRetryAsksAgainAfterNoAnswer(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		if calls == 1 {
			_, _ = w.Write([]byte(`{"answers":{"choice":{"choice":"invented"}}}`))
			return
		}
		_, _ = w.Write([]byte(`{"answers":{"choice":{"choice":"a","probabilities":{"a":1}}}}`))
	}))
	defer srv.Close()
	got, err := Retry(New(func() string { return "k" }, WithURL(srv.URL)), 3).Choose(t.Context(), nil, "q", map[string]string{"a": "", "b": ""})
	if err != nil || calls != 2 || got.Probabilities["a"] != 1 {
		t.Fatalf("calls=%d got=%#v err=%v", calls, got, err)
	}
	if c, err := Retry(nil, 3).Choose(t.Context(), nil, "q", map[string]string{"a": "", "b": ""}); !errors.Is(err, ErrNotConfigured) || c.Model != DefaultModel {
		t.Fatalf("nil client: %#v %v", c, err)
	}
}
