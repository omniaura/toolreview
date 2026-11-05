// Package toolreview decides which of an agent's tools must wait for a
// person's approval, by asking a decision model what each tool does.
//
// Tool catalogs label themselves. Composio marks deletes with destructiveHint
// and lookups with readOnlyHint; an MCP server may declare the same hints or
// none at all. Everything else is an unlabelled "write", and that bucket mixes
// a draft reply with a sent one, a private note with a public post, a board
// star with a payment. A word list cannot separate them: the words are shared.
//
// toolreview reads each tool once, as data (its name, title, description and
// arguments), and asks TypeSafe's Jev model which of four classes a call
// belongs to: read, write, outbound or destructive. The answer carries a
// calibrated probability per class, so a threshold decides what is adopted.
//
// Two rules keep it safe to use as a gate:
//
//   - A review only ever adds caution. The effective class is the more
//     consequential of the provider's own label and the review, so a tool the
//     provider marks destructive stays destructive whatever the model says,
//     and a description that argues "this is read-only" can at worst leave a
//     tool where its provider put it.
//   - A verdict is bound to the exact text the model read (Fingerprint) and
//     the policy that read it (Policy). Change either and the verdict is stale.
//
// Reviews are per tool, not per call. A catalog is reviewed once, when it
// appears or changes, and the verdicts are stored; the request path only looks
// them up. See Catalog for the cache, queue and offline fallback, and
// cmd/toolreview for reviewing a catalog and generating a Go manifest.
package toolreview

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"time"
)

// Policy names the prompt, labels and thresholds a verdict was produced
// under. Bump it when any of them changes; every stored verdict then reads as
// stale and is reviewed again.
const Policy = "toolreview/2026-11-05.1"

// Class is what one successful call of a tool does.
type Class string

const (
	// Unknown means nobody has said. It is never the result of a review.
	Unknown Class = ""
	// Read looks things up and changes nothing.
	Read Class = "read"
	// Write changes something only the person sees, and is easy to undo.
	Write Class = "write"
	// Outbound reaches other people or the outside world: sends, posts,
	// shares, invites, grants access, moves money.
	Outbound Class = "outbound"
	// Destructive deletes, overwrites or cancels in a way that is hard to undo.
	Destructive Class = "destructive"
)

// Classes lists every class a review can return, least consequential first.
var Classes = []Class{Read, Write, Outbound, Destructive}

// rank orders classes by consequence. Outbound and destructive both need
// approval; destructive ranks higher only so a merge is deterministic.
func (c Class) rank() int {
	switch c {
	case Read:
		return 1
	case Write:
		return 2
	case Outbound:
		return 3
	case Destructive:
		return 4
	default:
		return 0
	}
}

// Valid reports whether c is one of Classes.
func (c Class) Valid() bool { return c.rank() > 0 }

// RequiresApproval reports whether a call of this class must wait for a
// person: anything that leaves the person's own workspace or cannot be undone.
func (c Class) RequiresApproval() bool { return c == Outbound || c == Destructive }

// Max returns the more consequential of a and b.
func Max(a, b Class) Class {
	if b.rank() > a.rank() {
		return b
	}
	return a
}

// Tool is one entry of a catalog, as the model will read it.
type Tool struct {
	// Key identifies the tool within its source: a Composio slug, or an MCP
	// server prefix and tool name. It is not shown to the model.
	Key string `json:"key"`
	// Name is the tool's callable name.
	Name string `json:"name"`
	// Title is a human label, when the catalog has one.
	Title string `json:"title,omitempty"`
	// App is the product the tool acts on ("Outlook", "GitHub").
	App string `json:"app,omitempty"`
	// Description is the catalog's description of the tool.
	Description string `json:"description,omitempty"`
	// InputSchema is the JSON Schema of the tool's arguments.
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
	// Declared is the class the provider itself claims, from its behaviour
	// hints. It is not shown to the model, so the review is independent of it.
	Declared Class `json:"declared,omitempty"`
}

// Limits on what the model reads, so a large catalog costs a bounded amount
// and one tool with a pathological schema cannot dominate a review.
const (
	maxDescriptionRunes = 1500
	maxArgumentRunes    = 200
	maxArguments        = 40
	maxEnumValues       = 8
	maxSchemaDepth      = 2
)

// State is the JSON object the model reads for one tool. The provider's
// declared class and the catalog key are deliberately absent.
type State struct {
	Tool StateTool `json:"tool"`
}

// StateTool is the tool part of State.
type StateTool struct {
	Name        string     `json:"name"`
	Title       string     `json:"title,omitempty"`
	App         string     `json:"app,omitempty"`
	Description string     `json:"description,omitempty"`
	Arguments   []Argument `json:"arguments,omitempty"`
}

// Argument is one parameter of a tool, summarized from its JSON Schema.
type Argument struct {
	Name        string   `json:"name"`
	Type        string   `json:"type,omitempty"`
	Description string   `json:"description,omitempty"`
	Enum        []string `json:"enum,omitempty"`
	Required    bool     `json:"required,omitempty"`
}

// StateOf returns what the model reads for t.
func StateOf(t Tool) State {
	return State{Tool: StateTool{
		Name:        strings.TrimSpace(t.Name),
		Title:       strings.TrimSpace(t.Title),
		App:         strings.TrimSpace(t.App),
		Description: clip(strings.TrimSpace(t.Description), maxDescriptionRunes),
		Arguments:   SummarizeSchema(t.InputSchema),
	}}
}

// Fingerprint identifies exactly what the model reads for t. Two tools with
// the same fingerprint get the same review, so a verdict is reusable until the
// fingerprint changes.
func Fingerprint(t Tool) string {
	b, err := json.Marshal(StateOf(t))
	if err != nil {
		// StateOf holds only strings, bools and slices of them.
		panic("toolreview: encode state: " + err.Error())
	}
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:16])
}

// Verdict is the stored result of reviewing one tool.
type Verdict struct {
	ReviewedAt  time.Time `json:"reviewed_at,omitzero"`
	Key         string    `json:"key"`
	Fingerprint string    `json:"fingerprint"`
	Policy      string    `json:"policy"`
	Model       string    `json:"model,omitempty"`
	// Class is the review's adopted class, after thresholds.
	Class Class `json:"class"`
	// Declared is the provider's own class at review time.
	Declared Class `json:"declared,omitempty"`
	// Probabilities are the model's calibrated probabilities per class.
	Probabilities map[Class]float64 `json:"probabilities,omitempty"`
	// Cost is what the review cost upstream, in the provider's currency.
	Cost float64 `json:"cost,omitempty"`
	// Stale marks a verdict served for a tool whose fingerprint or policy no
	// longer matches. Catalog sets it; it is never stored.
	Stale bool `json:"-"`
}

// Effective is the class to enforce: the more consequential of the provider's
// declared class and the review. A review can raise a tool, never lower it.
func (v Verdict) Effective() Class { return Max(v.Declared, v.Class) }

// RequiresApproval reports whether calls of the reviewed tool must wait.
func (v Verdict) RequiresApproval() bool { return v.Effective().RequiresApproval() }

// Raised reports whether the review is more cautious than the provider: the
// tool was under-labelled, which is the case this package exists to catch.
func (v Verdict) Raised() bool { return v.Class.rank() > v.Declared.rank() && v.Declared != Unknown }

// Current reports whether v was produced for t as it is now, under this
// package's policy.
func (v Verdict) Current(t Tool) bool {
	return v.Policy == Policy && v.Fingerprint == Fingerprint(t)
}

// Risk is the probability the model gave to the classes that need approval.
func (v Verdict) Risk() float64 {
	return v.Probabilities[Outbound] + v.Probabilities[Destructive]
}

func clip(s string, limit int) string {
	if limit <= 0 {
		return ""
	}
	n := 0
	for i := range s {
		if n == limit {
			return strings.TrimSpace(s[:i]) + "…"
		}
		n++
	}
	return s
}
