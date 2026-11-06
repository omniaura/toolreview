package toolreview

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// SummarizeSchema flattens a tool's JSON Schema into the arguments a reviewer
// needs: name, type, a short description, a few enum values and whether it is
// required. Nested object properties are named with dots ("message.to") up to
// a fixed depth. Argument names are often the strongest signal of what a tool
// does (to, cc, recipients, amount, permission), and a summary reads them at a
// fraction of the cost of the full schema.
//
// An absent or unparseable schema has no arguments. Output order is stable:
// required arguments first, then by name.
func SummarizeSchema(raw json.RawMessage) []Argument {
	if len(raw) == 0 {
		return nil
	}
	var root schemaNode
	if err := json.Unmarshal(raw, &root); err != nil {
		return nil
	}
	var out []Argument
	collectArguments(&out, root, "", 0)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Required != out[j].Required {
			return out[i].Required
		}
		return out[i].Name < out[j].Name
	})
	if len(out) > maxArguments {
		out = out[:maxArguments]
	}
	return out
}

type schemaNode struct {
	Type        json.RawMessage       `json:"type"`
	Description string                `json:"description"`
	Title       string                `json:"title"`
	Properties  map[string]schemaNode `json:"properties"`
	Items       *schemaNode           `json:"items"`
	Required    []string              `json:"required"`
	Enum        []json.RawMessage     `json:"enum"`
	AnyOf       []schemaNode          `json:"anyOf"`
	OneOf       []schemaNode          `json:"oneOf"`
}

func collectArguments(out *[]Argument, node schemaNode, prefix string, depth int) {
	required := make(map[string]bool, len(node.Required))
	for _, name := range node.Required {
		required[name] = true
	}
	names := make([]string, 0, len(node.Properties))
	for name := range node.Properties {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if len(*out) >= maxArguments*2 {
			return
		}
		prop := node.Properties[name]
		full := name
		if prefix != "" {
			full = prefix + "." + name
		}
		desc := strings.TrimSpace(prop.Description)
		if desc == "" {
			desc = strings.TrimSpace(prop.Title)
		}
		*out = append(*out, Argument{
			Name:        clip(full, maxNameRunes),
			Type:        clip(schemaType(prop), 60),
			Description: clip(desc, maxArgumentRunes),
			Enum:        enumValues(prop.Enum),
			Required:    required[name] && prefix == "",
		})
		if depth+1 >= maxSchemaDepth {
			continue
		}
		child := prop
		if child.Items != nil {
			child = *child.Items
		}
		if len(child.Properties) > 0 {
			collectArguments(out, child, full, depth+1)
		}
	}
}

func schemaType(n schemaNode) string {
	if len(n.Type) > 0 {
		var one string
		if json.Unmarshal(n.Type, &one) == nil {
			if one == "array" && n.Items != nil {
				if inner := schemaType(*n.Items); inner != "" {
					return inner + "[]"
				}
			}
			return one
		}
		var many []string
		if json.Unmarshal(n.Type, &many) == nil {
			return strings.Join(many, "|")
		}
	}
	for _, alts := range [][]schemaNode{n.AnyOf, n.OneOf} {
		var types []string
		for i, alt := range alts {
			if i == maxEnumValues {
				break
			}
			if t := schemaType(alt); t != "" && t != "null" {
				types = append(types, t)
			}
		}
		if len(types) > 0 {
			return strings.Join(types, "|")
		}
	}
	return ""
}

func enumValues(raw []json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}
	out := make([]string, 0, min(len(raw), maxEnumValues))
	for _, v := range raw {
		if len(out) == maxEnumValues {
			break
		}
		var s string
		if json.Unmarshal(v, &s) == nil {
			out = append(out, clip(s, 40))
			continue
		}
		out = append(out, clip(fmt.Sprint(string(v)), 40))
	}
	return out
}
