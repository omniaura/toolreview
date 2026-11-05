package toolreview

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"go/parser"
	"go/token"
	"math"
	"strings"
	"sync"
	"testing"
	"time"
)

func p(read, write, outbound, destructive float64) map[Class]float64 {
	return map[Class]float64{Read: read, Write: write, Outbound: outbound, Destructive: destructive}
}

func TestAdopt(t *testing.T) {
	th := DefaultThresholds
	cases := []struct {
		name     string
		declared Class
		probs    map[Class]float64
		want     Class
	}{
		{"unlabelled send", Unknown, p(0.05, 0.1, 0.8, 0.05), Outbound},
		{"write that deletes", Write, p(0, 0.3, 0.1, 0.6), Destructive},
		{"risk split across both", Write, p(0.1, 0.35, 0.3, 0.25), Outbound},
		{"plain write", Write, p(0.1, 0.8, 0.05, 0.05), Write},
		{"unlabelled read", Unknown, p(0.9, 0.05, 0.03, 0.02), Read},
		{"unsure read is a write", Unknown, p(0.6, 0.3, 0.05, 0.05), Write},
		{"declared read stays read on weak doubt", Read, p(0.4, 0.3, 0.2, 0.1), Read},
		{"declared read that writes", Read, p(0.1, 0.75, 0.1, 0.05), Write},
		{"declared read that sends", Read, p(0.05, 0.1, 0.8, 0.05), Outbound},
		{"declared read, risk below override bar", Read, p(0.1, 0.2, 0.6, 0.1), Write},
	}
	for _, c := range cases {
		if got := th.Adopt(c.declared, c.probs); got != c.want {
			t.Errorf("%s: Adopt = %s, want %s", c.name, got, c.want)
		}
	}
}

func TestEffectiveNeverLowersTheProvider(t *testing.T) {
	for _, declared := range append([]Class{Unknown}, Classes...) {
		for _, reviewed := range Classes {
			v := Verdict{Declared: declared, Class: reviewed}
			if v.Effective().rank() < declared.rank() || v.Effective().rank() < reviewed.rank() {
				t.Fatalf("Effective(%s, %s) = %s", declared, reviewed, v.Effective())
			}
		}
	}
	if (Verdict{Declared: Destructive, Class: Read}).RequiresApproval() != true {
		t.Fatal("a provider's destructive label must hold whatever the review says")
	}
	if !(Verdict{Declared: Write, Class: Outbound}).Raised() || (Verdict{Declared: Unknown, Class: Outbound}).Raised() {
		t.Fatal("Raised is only for a provider label the review overrides")
	}
}

func TestFingerprintBindsWhatTheModelReads(t *testing.T) {
	base := Tool{Key: "OUTLOOK_REPLY_EMAIL", Name: "OUTLOOK_REPLY_EMAIL", Description: "Reply to an email",
		InputSchema: json.RawMessage(`{"properties":{"comment":{"type":"string"}}}`)}
	fp := Fingerprint(base)
	same := base
	same.Key, same.Declared = "other", Destructive
	if Fingerprint(same) != fp {
		t.Fatal("key and declared class are not shown to the model and must not change the fingerprint")
	}
	changed := base
	changed.Description = "Reply to an email and cc the whole company"
	if Fingerprint(changed) == fp {
		t.Fatal("a changed description must change the fingerprint")
	}
	changed = base
	changed.InputSchema = json.RawMessage(`{"properties":{"comment":{"type":"string"},"cc":{"type":"string"}}}`)
	if Fingerprint(changed) == fp {
		t.Fatal("a new argument must change the fingerprint")
	}
	v := Verdict{Fingerprint: fp, Policy: Policy}
	if !v.Current(base) || v.Current(changed) {
		t.Fatal("Current must follow the fingerprint")
	}
	if (Verdict{Fingerprint: fp, Policy: "old"}).Current(base) {
		t.Fatal("a verdict from another policy is not current")
	}
}

func TestSummarizeSchema(t *testing.T) {
	raw := json.RawMessage(`{
		"type": "object",
		"required": ["to", "body"],
		"properties": {
			"to": {"type": "array", "items": {"type": "string"}, "description": "Recipients"},
			"body": {"type": "string"},
			"importance": {"type": "string", "enum": ["low", "normal", "high"]},
			"amount": {"anyOf": [{"type": "number"}, {"type": "null"}]},
			"message": {"type": "object", "properties": {"subject": {"type": "string", "title": "Subject line"},
				"deep": {"type": "object", "properties": {"hidden": {"type": "string"}}}}}
		}
	}`)
	args := SummarizeSchema(raw)
	names := make([]string, len(args))
	for i, a := range args {
		names[i] = a.Name
	}
	want := "body,to,amount,importance,message,message.deep,message.subject"
	if got := strings.Join(names, ","); got != want {
		t.Fatalf("arguments = %s, want %s", got, want)
	}
	byName := map[string]Argument{}
	for _, a := range args {
		byName[a.Name] = a
	}
	if byName["to"].Type != "string[]" || !byName["to"].Required || byName["to"].Description != "Recipients" {
		t.Fatalf("to = %#v", byName["to"])
	}
	if byName["amount"].Type != "number" || strings.Join(byName["importance"].Enum, ",") != "low,normal,high" {
		t.Fatalf("amount/importance = %#v %#v", byName["amount"], byName["importance"])
	}
	if byName["message.subject"].Description != "Subject line" || byName["message.subject"].Required {
		t.Fatalf("nested = %#v", byName["message.subject"])
	}
	if SummarizeSchema(json.RawMessage(`not json`)) != nil || SummarizeSchema(nil) != nil {
		t.Fatal("an unreadable schema has no arguments")
	}
}

func TestStateIsBounded(t *testing.T) {
	long := strings.Repeat("é", maxDescriptionRunes+50)
	s := StateOf(Tool{Name: "x", Description: long})
	if n := len([]rune(s.Tool.Description)); n != maxDescriptionRunes+1 {
		t.Fatalf("description runes = %d", n)
	}
	props := map[string]any{}
	for i := range 100 {
		props[strings.Repeat("a", i+1)] = map[string]any{"type": "string"}
	}
	raw, _ := json.Marshal(map[string]any{"properties": props})
	if n := len(SummarizeSchema(raw)); n != maxArguments {
		t.Fatalf("arguments = %d, want %d", n, maxArguments)
	}
}

type fakeChooser struct {
	mu     sync.Mutex
	calls  int
	states []State
	answer func(State) (map[string]float64, error)
}

func (f *fakeChooser) Choose(_ context.Context, state any, instructions string, criteria map[string]string) (Choice, error) {
	f.mu.Lock()
	f.calls++
	f.states = append(f.states, state.(State))
	f.mu.Unlock()
	if instructions != Instructions || len(criteria) != 4 {
		return Choice{}, errors.New("unexpected question")
	}
	probs, err := f.answer(state.(State))
	return Choice{Probabilities: probs, Model: "fake", Cost: 0.001}, err
}

func sendy(State) (map[string]float64, error) {
	return map[string]float64{"read": 0.02, "write": 0.08, "outbound": 0.85, "destructive": 0.05}, nil
}

func TestReviewerReview(t *testing.T) {
	f := &fakeChooser{answer: sendy}
	r := NewReviewer(f)
	r.Now = func() time.Time { return time.Date(2026, 11, 5, 0, 0, 0, 0, time.UTC) }
	tool := Tool{Key: "K", Name: "OUTLOOK_REPLY_EMAIL", App: "Outlook", Description: "Reply", Declared: Write}
	v, err := r.Review(t.Context(), tool)
	if err != nil {
		t.Fatal(err)
	}
	if v.Class != Outbound || !v.RequiresApproval() || !v.Raised() || v.Model != "fake" || v.Cost != 0.001 || !v.Current(tool) {
		t.Fatalf("verdict = %#v", v)
	}
	if f.states[0].Tool.Name != "OUTLOOK_REPLY_EMAIL" || f.states[0].Tool.App != "Outlook" {
		t.Fatalf("state = %#v", f.states[0])
	}
	b, _ := json.Marshal(f.states[0])
	if strings.Contains(string(b), "declared") || strings.Contains(string(b), `"K"`) {
		t.Fatalf("the provider's label and the key must not reach the model: %s", b)
	}
}

func TestReviewerRejectsNonProbabilities(t *testing.T) {
	for _, bad := range []map[string]float64{
		{"read": math.NaN(), "write": 0.5},
		{"read": 1.5},
		{"read": -0.1, "write": 0.5},
		{"read": 0.9, "write": 0.9},
		{},
		{"unknown_label": 1},
	} {
		f := &fakeChooser{answer: func(State) (map[string]float64, error) { return bad, nil }}
		if _, err := NewReviewer(f).Review(t.Context(), Tool{Key: "k", Name: "n"}); !errors.Is(err, ErrInvalidAnswer) {
			t.Fatalf("%v: err = %v", bad, err)
		}
	}
}

type memStore struct {
	mu    sync.Mutex
	rows  map[string]Verdict
	loads int
	fail  error
}

func newMemStore() *memStore { return &memStore{rows: map[string]Verdict{}} }

func (m *memStore) Load(_ context.Context, source string, keys []string) (map[string]Verdict, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.loads++
	if m.fail != nil {
		return nil, m.fail
	}
	out := map[string]Verdict{}
	for _, k := range keys {
		if v, ok := m.rows[source+"/"+k]; ok {
			out[k] = v
		}
	}
	return out, nil
}

func (m *memStore) Save(_ context.Context, source string, v Verdict) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.rows[source+"/"+v.Key] = v
	return nil
}

func TestCatalogLookupQueueAndDrain(t *testing.T) {
	f := &fakeChooser{answer: sendy}
	store := newMemStore()
	now := time.Date(2026, 11, 5, 0, 0, 0, 0, time.UTC)
	reply := Tool{Key: "OUTLOOK_REPLY_EMAIL", Name: "OUTLOOK_REPLY_EMAIL", Description: "Reply", Declared: Write}
	draft := Tool{Key: "OUTLOOK_CREATE_DRAFT", Name: "OUTLOOK_CREATE_DRAFT", Description: "Draft", Declared: Write}
	fallback := ManifestOf([]Verdict{{Key: draft.Key, Fingerprint: Fingerprint(draft), Class: Write, Declared: Write}})
	c := NewCatalog(NewReviewer(f), store, WithFallback("composio", fallback), WithClock(func() time.Time { return now }))

	got := c.Verdicts(t.Context(), "composio", []Tool{reply, draft})
	if _, ok := got[reply.Key]; ok || got[draft.Key].Class != Write || got[draft.Key].Stale {
		t.Fatalf("first lookup = %#v", got)
	}
	if c.Pending() != 1 || f.calls != 0 {
		t.Fatalf("pending=%d calls=%d: lookups must not call the model", c.Pending(), f.calls)
	}

	stats := c.Drain(t.Context(), 10, 2)
	if stats.Reviewed != 1 || stats.Raised != 1 || c.Pending() != 0 || f.calls != 1 {
		t.Fatalf("drain = %#v pending=%d calls=%d", stats, c.Pending(), f.calls)
	}
	if store.rows["composio/"+reply.Key].Class != Outbound {
		t.Fatal("drain must persist the verdict")
	}
	got = c.Verdicts(t.Context(), "composio", []Tool{reply})
	if !got[reply.Key].RequiresApproval() || got[reply.Key].Stale {
		t.Fatalf("after drain = %#v", got)
	}

	// The provider changes the tool: the old verdict is still served, marked
	// stale, and the tool is queued again.
	changed := reply
	changed.Description = "Reply, now with attachments"
	got = c.Verdicts(t.Context(), "composio", []Tool{changed})
	if !got[reply.Key].Stale || !got[reply.Key].RequiresApproval() || c.Pending() != 1 {
		t.Fatalf("changed tool = %#v pending=%d", got, c.Pending())
	}

	// Another replica reviewed it meanwhile: drain takes the stored verdict.
	other := NewCatalog(NewReviewer(f), store)
	if _, err := other.Review(t.Context(), "composio", changed); err != nil {
		t.Fatal(err)
	}
	calls := f.calls
	if stats := c.Drain(t.Context(), 10, 1); stats.Current != 1 || stats.Reviewed != 0 || f.calls != calls {
		t.Fatalf("drain should reuse the stored review: %#v calls=%d→%d", stats, calls, f.calls)
	}
}

func TestCatalogDegradesWhenTheStoreFails(t *testing.T) {
	store := newMemStore()
	store.fail = errors.New("db down")
	tool := Tool{Key: "A", Name: "A"}
	var reported int
	c := NewCatalog(nil, store, WithFallback("s", ManifestOf([]Verdict{{Key: "A", Fingerprint: "old", Class: Outbound}})),
		WithErrorHandler(func(string, string, error) { reported++ }))
	got := c.Verdicts(t.Context(), "s", []Tool{tool})
	if !got["A"].Stale || !got["A"].RequiresApproval() || reported != 1 {
		t.Fatalf("got %#v reported=%d", got, reported)
	}
	// The failed key is not re-queried on every lookup.
	c.Verdicts(t.Context(), "s", []Tool{tool})
	if store.loads != 1 {
		t.Fatalf("store loads = %d", store.loads)
	}
	if _, err := c.Review(t.Context(), "s", tool); err == nil {
		t.Fatal("a catalog without a reviewer cannot review")
	}
}

func TestDrainRetriesThenGivesUp(t *testing.T) {
	f := &fakeChooser{answer: func(State) (map[string]float64, error) { return nil, errors.New("upstream 503") }}
	c := NewCatalog(NewReviewer(f), nil)
	c.Enqueue("s", Tool{Key: "A", Name: "A"})
	for i := range maxAttempts {
		stats := c.Drain(t.Context(), 10, 1)
		if stats.Failed != 1 {
			t.Fatalf("attempt %d: %#v", i, stats)
		}
	}
	if c.Pending() != 0 || f.calls != maxAttempts {
		t.Fatalf("pending=%d calls=%d", c.Pending(), f.calls)
	}
}

func TestWriteGo(t *testing.T) {
	m := ManifestOf([]Verdict{
		{Key: "B_SEND", Fingerprint: "fb", Policy: Policy, Model: "typesafe/jev-1.13", Class: Outbound, Declared: Write,
			Probabilities: p(0, 0.1, 0.8766, 0.0234)},
		{Key: "A_LIST", Fingerprint: "fa", Class: Read, Declared: Read},
	})
	var b bytes.Buffer
	if err := WriteGo(&b, "reviews", "composioReviews", "Composio", m); err != nil {
		t.Fatal(err)
	}
	src := b.String()
	if _, err := parser.ParseFile(token.NewFileSet(), "gen.go", src, 0); err != nil {
		t.Fatalf("generated source does not parse: %v\n%s", err, src)
	}
	for _, want := range []string{
		"// Code generated by toolreview gen; DO NOT EDIT.",
		`"A_LIST": {"fa", toolreview.Read, toolreview.Read, 0},`,
		`"B_SEND": {"fb", toolreview.Outbound, toolreview.Write, 0.9},`,
		`Model:  "typesafe/jev-1.13"`,
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("missing %q in\n%s", want, src)
		}
	}
	if strings.Index(src, "A_LIST") > strings.Index(src, "B_SEND") {
		t.Fatal("entries must be sorted for stable diffs")
	}
}
