package toolreview

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"go/parser"
	"go/token"
	"math"
	"sort"
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
		{"a read-only hint does not raise the hold bar", Read, p(0.1, 0.2, 0.6, 0.1), Outbound},
	}
	for _, c := range cases {
		if got := th.Adopt(c.declared, c.probs); got != c.want {
			t.Errorf("%s: Adopt = %s, want %s", c.name, got, c.want)
		}
	}
}

// A review can only raise a tool. Over a grid of answers: the enforced class
// is never below the provider's label, and a provider's read-only label never
// yields fewer holds than no label at all.
func TestReviewNeverLowersCaution(t *testing.T) {
	steps := []float64{0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.8, 1}
	checked := 0
	for _, r := range steps {
		for _, w := range steps {
			for _, o := range steps {
				d := 1 - r - w - o
				if d < -1e-9 {
					continue
				}
				probs := map[string]float64{"read": r, "write": w, "outbound": o, "destructive": max(d, 0)}
				f := &fakeChooser{answer: func(State) (map[string]float64, error) { return probs, nil }}
				rev := NewReviewer(f)
				held := map[Class]bool{}
				for _, declared := range append([]Class{Unknown}, Classes...) {
					v, err := rev.Review(t.Context(), Tool{Key: "k", Name: "n", Declared: declared})
					if err != nil {
						t.Fatal(err)
					}
					if v.Effective().rank() < declared.rank() {
						t.Fatalf("%v declared %s: effective %s", probs, declared, v.Effective())
					}
					held[declared] = v.RequiresApproval()
					checked++
				}
				if held[Unknown] && !held[Read] {
					t.Fatalf("%v: a read-only hint released a hold", probs)
				}
			}
		}
	}
	if checked < 100 {
		t.Fatalf("grid covered only %d reviews", checked)
	}
}

// policyDigest pins what Policy names. If this test fails, the prompt,
// labels or thresholds changed: bump Policy, then update the digest.
const policyDigest = "f782f3ee71f0a814"

func TestPolicyNamesThePrompt(t *testing.T) {
	b, _ := json.Marshal(struct {
		Instructions string
		Criteria     map[string]string
		Thresholds   Thresholds
	}{Instructions, Criteria(), DefaultThresholds})
	sum := sha256.Sum256(b)
	if got := hex.EncodeToString(sum[:8]); got != policyDigest {
		t.Fatalf("prompt digest %s != %s: bump Policy (%s) and update policyDigest", got, policyDigest, Policy)
	}
	c := Criteria()
	c["read"] = "mutated"
	if Criteria()["read"] == "mutated" {
		t.Fatal("Criteria must hand out a copy")
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
	fallback := ManifestOf([]Verdict{{Key: draft.Key, Fingerprint: Fingerprint(draft), Policy: Policy, Class: Write, Declared: Write}})
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

func TestFailingReviewBacksOff(t *testing.T) {
	f := &fakeChooser{answer: func(State) (map[string]float64, error) { return nil, errors.New("upstream 503") }}
	now := time.Date(2026, 11, 5, 0, 0, 0, 0, time.UTC)
	c := NewCatalog(NewReviewer(f), nil, WithClock(func() time.Time { return now }))
	tool := Tool{Key: "A", Name: "A"}
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		c.Verdicts(t.Context(), "s", []Tool{tool})
		if stats := c.Drain(t.Context(), 10, 1); stats.Failed != 1 {
			t.Fatalf("attempt %d: %#v", attempt, stats)
		}
		// The request path cannot re-queue it inside the backoff.
		c.Verdicts(t.Context(), "s", []Tool{tool})
		c.Enqueue("s", tool)
		if c.Pending() != 0 {
			t.Fatalf("attempt %d: re-queued inside its backoff", attempt)
		}
		now = now.Add(time.Duration(attempt)*time.Minute + time.Second)
	}
	if f.calls != maxAttempts {
		t.Fatalf("calls = %d", f.calls)
	}
	// The request path honours the backoff too: no model call.
	if _, err := c.Review(t.Context(), "s", tool); !errors.Is(err, ErrBackoff) || f.calls != maxAttempts {
		t.Fatalf("Review during backoff: err=%v calls=%d", err, f.calls)
	}
	c.Enqueue("s", tool)
	if c.Pending() != 0 {
		t.Fatal("after maxAttempts the tool waits failedBackoff")
	}
	changed := tool
	changed.Description = "now different"
	c.Enqueue("s", changed)
	if c.Pending() != 1 {
		t.Fatal("a changed tool is retried at once")
	}
}

func TestCancelledReviewIsNotAFailure(t *testing.T) {
	f := &fakeChooser{answer: func(State) (map[string]float64, error) { return nil, context.DeadlineExceeded }}
	c := NewCatalog(NewReviewer(f), nil)
	tool := Tool{Key: "A", Name: "A"}
	for range maxAttempts + 1 {
		c.Enqueue("s", tool)
		c.Drain(t.Context(), 10, 1)
		if _, err := c.Review(t.Context(), "s", tool); err == nil {
			t.Fatal("review should fail")
		}
	}
	c.mu.Lock()
	n := len(c.failures)
	c.mu.Unlock()
	if n != 0 || c.Pending() != 1 {
		t.Fatalf("a cancelled review was counted as a failure: failures=%d pending=%d", n, c.Pending())
	}
}

func TestStaleStoreVerdictDoesNotReplaceCachedHold(t *testing.T) {
	store := newMemStore()
	tool := Tool{Key: "A", Name: "A", Description: "v2"}
	now := time.Date(2026, 11, 5, 0, 0, 0, 0, time.UTC)
	c := NewCatalog(nil, store, WithClock(func() time.Time { return now }),
		WithFallback("s", ManifestOf([]Verdict{{Key: "A", Fingerprint: "v1", Policy: Policy, Class: Outbound}})))
	c.mu.Lock()
	c.remember(catalogKey{"s", "A"}, Verdict{Key: "A", Fingerprint: "v1", Policy: Policy, Class: Outbound})
	c.mu.Unlock()
	store.rows["s/A"] = Verdict{Key: "A", Fingerprint: "v0", Policy: Policy, Class: Write}
	if got := c.Verdicts(t.Context(), "s", []Tool{tool}); !got["A"].RequiresApproval() || !got["A"].Stale {
		t.Fatalf("got %#v", got)
	}
}

func TestEvictionKeepsHolds(t *testing.T) {
	c := NewCatalog(nil, nil, WithLimits(0, 8))
	c.mu.Lock()
	for i := range 8 {
		cls := Write
		if i%4 == 0 {
			cls = Outbound
		}
		c.remember(catalogKey{"s", string(rune('a' + i))}, Verdict{Class: cls})
	}
	c.remember(catalogKey{"s", "new"}, Verdict{Class: Read})
	holds := 0
	for _, v := range c.cache {
		if v.Class == Outbound {
			holds++
		}
	}
	c.mu.Unlock()
	if holds != 2 || len(c.cache) > 8 {
		t.Fatalf("holds=%d size=%d", holds, len(c.cache))
	}
}

func TestManifestKeepsTheVerdictsPolicy(t *testing.T) {
	old := ManifestOf([]Verdict{{Key: "A", Policy: "toolreview/old", Class: Outbound}})
	if old.Policy != "toolreview/old" {
		t.Fatalf("policy = %q", old.Policy)
	}
	mixed := ManifestOf([]Verdict{
		{Key: "A", Policy: Policy, Fingerprint: Fingerprint(Tool{Name: "A"}), Class: Outbound},
		{Key: "C", Policy: Policy, Fingerprint: Fingerprint(Tool{Name: "C"}), Class: Read},
		{Key: "B", Policy: "toolreview/old", Fingerprint: Fingerprint(Tool{Name: "B"}), Class: Write},
	})
	if mixed.Policy != Policy || mixed.Entries["B"].Policy != "toolreview/old" || mixed.Entries["A"].Policy != "" {
		t.Fatalf("mixed = %#v", mixed)
	}
	a, _ := mixed.verdict("A")
	b, _ := mixed.verdict("B")
	if !a.Current(Tool{Name: "A"}) || b.Current(Tool{Name: "B"}) {
		t.Fatal("staleness must be per entry: current entries stay current, old ones read stale")
	}
	var src bytes.Buffer
	if err := WriteGo(&src, "x", "m", "test", mixed); err != nil || !strings.Contains(src.String(), `"B": {Policy: "toolreview/old", `) {
		t.Fatalf("generated per-entry policy missing: %v\n%s", err, src.String())
	}
}

func TestWriteGo(t *testing.T) {
	m := ManifestOf([]Verdict{
		{Key: "B_SEND", Fingerprint: "fb", Policy: Policy, Model: "typesafe/jev-1.13", Class: Outbound, Declared: Write,
			Probabilities: p(0, 0.1, 0.8766, 0.0234)},
		{Key: "A_LIST", Fingerprint: "fa", Policy: Policy, Model: "typesafe/jev-1.13", Class: Read, Declared: Read},
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
		`"A_LIST": {Fingerprint: "fa", Class: toolreview.Read, Declared: toolreview.Read, Risk: 0},`,
		`"B_SEND": {Fingerprint: "fb", Class: toolreview.Outbound, Declared: toolreview.Write, Risk: 0.9},`,
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

func TestManifestHolds(t *testing.T) {
	m := ManifestOf([]Verdict{
		{Key: "SEND", Class: Outbound, Declared: Write},
		{Key: "DELETE", Class: Destructive, Declared: Destructive},
		{Key: "LIST", Class: Read, Declared: Read},
		{Key: "READ_THAT_POSTS", Class: Outbound, Declared: Read},
		{Key: "READ_THAT_WRITES", Class: Write, Declared: Read},
		{Key: "NEW_SEND", Class: Outbound},
	})
	h := m.Holds()
	got := make([]string, 0, len(h.Entries))
	for k := range h.Entries {
		got = append(got, k)
	}
	sort.Strings(got)
	if strings.Join(got, ",") != "NEW_SEND,READ_THAT_POSTS,SEND" || h.Policy != m.Policy {
		t.Fatalf("holds = %v", got)
	}
}
