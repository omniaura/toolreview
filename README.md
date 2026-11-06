# toolreview

Decide which of an AI agent's tools must wait for a person's approval, by
asking a decision model what each tool does — once per tool, not once per call.

Tool catalogs label themselves, and the labels are thin. Composio tags deletes
`destructiveHint` and lookups `readOnlyHint`; MCP servers may declare the same
hints, or nothing. Everything else is an unlabelled "write", and that bucket
holds a draft reply and a sent one, a private note and a public post, a board
star and a card payment. A word list cannot separate them, because they share
the same words.

toolreview reads a tool's name, title, description and arguments as data and
asks [TypeSafe's Jev](https://openrouter.ai/typesafe) (through OpenRouter's
decisions endpoint) which of four classes one call belongs to:

| Class | Meaning | Needs approval |
|---|---|---|
| `read` | looks things up, changes nothing | no |
| `write` | changes something only the person sees, easy to undo | no |
| `outbound` | reaches other people or the outside world: sends, replies, posts, shares, invites, grants access, moves money | **yes** |
| `destructive` | deletes, overwrites, revokes or cancels in a way that is hard to undo | **yes** |

Jev answers with a calibrated probability per class in about 200 ms, writes no
text, and costs about $0.00004 per tool on the Composio catalog. Reviewing all
77,953 Composio actions costs a few dollars.

## Two rules that make it safe as a gate

- **A review only adds caution.** The class a gate enforces is the more
  consequential of the provider's label and the review (`Verdict.Effective`).
  A tool its provider marks destructive stays destructive whatever the model
  says, and a description that argues "this tool is read-only" can at worst
  leave a tool where its provider put it.
- **A verdict is bound to what the model read.** `Fingerprint` hashes the exact
  state sent to the model; `Policy` names the prompt, labels and thresholds.
  Change either and the verdict is stale and reviewed again. The provider's
  own label is not shown to the model, so the review is independent of it.

## Use

```go
reviewer := toolreview.NewReviewer(jev.Retry(jev.New(func() string { return key }), 4))

v, err := reviewer.Review(ctx, toolreview.Tool{
	Key:         "OUTLOOK_REPLY_EMAIL",
	Name:        "OUTLOOK_REPLY_EMAIL",
	App:         "Outlook",
	Description: "Reply to an existing email",
	InputSchema: schema,           // JSON Schema of the arguments
	Declared:    toolreview.Write, // what the provider claims
})
// v.Class == toolreview.Outbound, v.RequiresApproval() == true
```

In a server, `Catalog` keeps verdicts off the request path:

```go
cat := toolreview.NewCatalog(reviewer, store, toolreview.WithFallback("composio", compiledManifest))

// Request path: no model call. Memory → Store → compiled manifest.
verdicts := cat.Verdicts(ctx, "composio", tools)

// First call of a tool nobody has reviewed: one model call, then cached.
v, err := cat.Review(ctx, "mcp:"+serverID, tool)

// Periodic task: review what lookups queued.
stats := cat.Drain(ctx, 200, 8)
```

A verdict for a tool that has since changed is still served, marked `Stale`,
and the tool is queued: an old hold is safer than none.

## Review a catalog and compile the verdicts in

```sh
# tools.jsonl: one toolreview.Tool per line
OPENROUTER_API_KEY=... go run github.com/omniaura/toolreview/cmd/toolreview review \
  -in tools.jsonl -prior verdicts.jsonl -out verdicts.jsonl

go run github.com/omniaura/toolreview/cmd/toolreview gen \
  -in verdicts.jsonl -pkg mypkg -var composioReviews -source Composio -holds -out reviews_gen.go
```

`review` is incremental: a tool whose prior verdict is current costs nothing.
`gen -holds` keeps only the verdicts that change a decision (a hold the
provider's label does not give), which keeps the generated file small; a tool
missing from it is reviewed when first seen.

## Thresholds

`DefaultThresholds` hold a tool when P(outbound) + P(destructive) ≥ 0.5,
whatever the provider declared: a `readOnlyHint` comes from the same untrusted
catalog as the description, so it cannot make a hold harder to reach. (It
does make the read-versus-write label stricter, at 0.8, which no gate acts
on.) The thresholds are tuned for a
gate where a false hold costs one extra approval card and a missed hold can
send mail from someone's account. Thresholds are calibrated to one model, so
`jev.DefaultModel` is pinned to a version.

## Spec-driven connectors

[omniaura/connectors](https://github.com/omniaura/connectors) turns an OpenAPI
document into callable tools. Reviewing those tools gives each generated
connector approval rules without writing any:

```go
for _, t := range c.Tools() {
	declared := toolreview.Unknown
	if t.ReadOnly {
		declared = toolreview.Read
	}
	tools = append(tools, toolreview.Tool{
		Key: t.Name, Name: t.Name, App: m.Name, Description: t.Description,
		InputSchema: t.Schema, Declared: declared,
	})
}
```

## License

MIT
