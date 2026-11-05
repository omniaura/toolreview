// Command toolreview reviews a tool catalog with Jev and generates a Go
// manifest of the verdicts.
//
//	toolreview review -in tools.json -prior verdicts.jsonl -out verdicts.jsonl
//	toolreview gen -in verdicts.jsonl -pkg mypkg -var composioReviews -source composio -out reviews_gen.go
//
// review reads OPENROUTER_API_KEY from the environment. It is incremental: a
// tool whose prior verdict is current (same fingerprint, same policy) is
// copied without a model call, so re-running after a catalog refresh costs
// only the tools that changed.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/omniaura/toolreview"
	"github.com/omniaura/toolreview/jev"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	var err error
	switch os.Args[1] {
	case "review":
		err = review(ctx, os.Args[2:])
	case "gen":
		err = gen(os.Args[2:])
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "toolreview:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: toolreview review|gen [flags]; see -h of each")
	os.Exit(2)
}

func review(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("review", flag.ExitOnError)
	in := fs.String("in", "", "tools: a JSON array or JSON lines of toolreview.Tool")
	prior := fs.String("prior", "", "previous verdicts (JSON lines); current ones are reused")
	out := fs.String("out", "", "verdicts output (JSON lines)")
	concurrency := fs.Int("concurrency", 8, "model calls in flight")
	limit := fs.Int("limit", 0, "review at most this many tools that need it (0 = all)")
	model := fs.String("model", jev.DefaultModel, "decision model")
	maxCost := fs.Float64("max-cost", 0, "stop starting reviews once this much has been spent (0 = no cap)")
	_ = fs.Parse(args)
	if *in == "" || *out == "" {
		return errors.New("review needs -in and -out")
	}
	key := strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY"))
	if key == "" {
		return errors.New("OPENROUTER_API_KEY is not set")
	}
	tools, err := readTools(*in)
	if err != nil {
		return err
	}
	previous := map[string]toolreview.Verdict{}
	if *prior != "" {
		vs, err := readVerdicts(*prior)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		for _, v := range vs {
			previous[v.Key] = v
		}
	}

	results := make(map[string]toolreview.Verdict, len(tools))
	var todo []toolreview.Tool
	for _, t := range tools {
		if v, ok := previous[t.Key]; ok && v.Current(t) {
			v.Declared = t.Declared
			results[t.Key] = v
			continue
		}
		todo = append(todo, t)
	}
	if *limit > 0 && len(todo) > *limit {
		todo = todo[:*limit]
	}
	fmt.Fprintf(os.Stderr, "%d tools, %d reused, %d to review with %s\n", len(tools), len(results), len(todo), *model)

	client := jev.New(func() string { return key }, jev.WithModel(*model),
		jev.WithHeader("HTTP-Referer", "https://github.com/omniaura/toolreview"), jev.WithHeader("X-Title", "toolreview"))
	reviewer := toolreview.NewReviewer(retrying{client})
	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	var mu sync.Mutex
	var spent float64
	var done, failed int
	started := time.Now()
	reviewer.ReviewAll(runCtx, todo, *concurrency, func(t toolreview.Tool, v toolreview.Verdict, err error) {
		mu.Lock()
		defer mu.Unlock()
		spent += v.Cost
		done++
		if err != nil {
			failed++
			fmt.Fprintf(os.Stderr, "  %s: %v\n", t.Key, err)
		} else {
			results[t.Key] = v
		}
		if done%250 == 0 {
			fmt.Fprintf(os.Stderr, "  %d/%d reviewed, %d failed, $%.4f, %s\n", done, len(todo), failed, spent, time.Since(started).Round(time.Second))
		}
		if *maxCost > 0 && spent >= *maxCost {
			cancel()
		}
	})

	verdicts := make([]toolreview.Verdict, 0, len(results))
	for _, v := range results {
		verdicts = append(verdicts, v)
	}
	if err := writeVerdicts(*out, verdicts); err != nil {
		return err
	}
	summarize(os.Stderr, tools, results)
	fmt.Fprintf(os.Stderr, "reviewed %d (%d failed) in %s, cost $%.4f\n", done, failed, time.Since(started).Round(time.Second), spent)
	if failed > 0 {
		return fmt.Errorf("%d reviews failed; re-run with -prior %s to retry only those", failed, *out)
	}
	return nil
}

// retrying retries a rate-limited or failed upstream call a few times.
type retrying struct{ c *jev.Client }

func (r retrying) Choose(ctx context.Context, state any, instructions string, criteria map[string]string) (toolreview.Choice, error) {
	var last error
	var cost float64
	for attempt := range 4 {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return toolreview.Choice{Cost: cost}, ctx.Err()
			case <-time.After(time.Duration(attempt*attempt) * time.Second):
			}
		}
		choice, err := r.c.Choose(ctx, state, instructions, criteria)
		cost += choice.Cost
		if err == nil {
			choice.Cost = cost
			return choice, nil
		}
		last = err
		var httpErr *jev.HTTPError
		if errors.As(err, &httpErr) && !httpErr.Retryable() {
			break
		}
	}
	return toolreview.Choice{Model: r.c.Model(), Cost: cost}, last
}

func summarize(w io.Writer, tools []toolreview.Tool, results map[string]toolreview.Verdict) {
	type pair struct{ declared, effective toolreview.Class }
	counts := map[pair]int{}
	for _, t := range tools {
		v, ok := results[t.Key]
		if !ok {
			continue
		}
		v.Declared = t.Declared
		counts[pair{t.Declared, v.Effective()}]++
	}
	keys := make([]pair, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].declared != keys[j].declared {
			return keys[i].declared < keys[j].declared
		}
		return keys[i].effective < keys[j].effective
	})
	fmt.Fprintln(w, "declared -> effective")
	for _, k := range keys {
		d := string(k.declared)
		if d == "" {
			d = "(none)"
		}
		fmt.Fprintf(w, "  %-12s -> %-12s %d\n", d, k.effective, counts[k])
	}
}

func gen(args []string) error {
	fs := flag.NewFlagSet("gen", flag.ExitOnError)
	in := fs.String("in", "", "verdicts (JSON lines)")
	out := fs.String("out", "", "Go file to write")
	pkg := fs.String("pkg", "", "package name")
	name := fs.String("var", "reviews", "variable name")
	source := fs.String("source", "tool", "what the verdicts are of, for the doc comment")
	_ = fs.Parse(args)
	if *in == "" || *out == "" || *pkg == "" {
		return errors.New("gen needs -in, -out and -pkg")
	}
	verdicts, err := readVerdicts(*in)
	if err != nil {
		return err
	}
	var b bytes.Buffer
	if err := toolreview.WriteGo(&b, *pkg, *name, *source, toolreview.ManifestOf(verdicts)); err != nil {
		return err
	}
	return os.WriteFile(*out, b.Bytes(), 0o644)
}

func readTools(path string) ([]toolreview.Tool, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var tools []toolreview.Tool
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) > 0 && trimmed[0] == '[' {
		if err := json.Unmarshal(trimmed, &tools); err != nil {
			return nil, fmt.Errorf("decode %s: %w", path, err)
		}
	} else {
		sc := bufio.NewScanner(bytes.NewReader(raw))
		sc.Buffer(make([]byte, 0, 1<<20), 64<<20)
		for sc.Scan() {
			line := bytes.TrimSpace(sc.Bytes())
			if len(line) == 0 {
				continue
			}
			var t toolreview.Tool
			if err := json.Unmarshal(line, &t); err != nil {
				return nil, fmt.Errorf("decode %s: %w", path, err)
			}
			tools = append(tools, t)
		}
		if err := sc.Err(); err != nil {
			return nil, err
		}
	}
	seen := make(map[string]bool, len(tools))
	for _, t := range tools {
		if t.Key == "" || t.Name == "" {
			return nil, fmt.Errorf("%s: every tool needs a key and a name", path)
		}
		if seen[t.Key] {
			return nil, fmt.Errorf("%s: duplicate key %q", path, t.Key)
		}
		seen[t.Key] = true
	}
	return tools, nil
}

func readVerdicts(path string) ([]toolreview.Verdict, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []toolreview.Verdict
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 1<<20), 16<<20)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var v toolreview.Verdict
		if err := json.Unmarshal(line, &v); err != nil {
			return nil, fmt.Errorf("decode %s: %w", path, err)
		}
		out = append(out, v)
	}
	return out, sc.Err()
}

func writeVerdicts(path string, verdicts []toolreview.Verdict) error {
	sort.Slice(verdicts, func(i, j int) bool { return verdicts[i].Key < verdicts[j].Key })
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	for _, v := range verdicts {
		if err := enc.Encode(v); err != nil {
			return err
		}
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b.Bytes(), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
