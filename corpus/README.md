# Corpus format

Everything the demo trains on lives here. Nothing is downloaded at run time, so
`python run_demo.py` works offline and produces byte-identical shards on any
machine.

## File layout

```
corpus/
  sources.json          provenance, licence and tier metadata per source file
  general_web/*.txt     lane files
  code/*.txt
  math_science/*.txt
  indic/*.txt
  agentic/*.txt
  reasoning/*.txt
  validation/*.txt      readable during training for eval, never gradient-bearing
  test/*.txt            benchmark data, never_train, plus canary strings
```

## Document format

A lane file holds several documents separated by a line containing exactly
`===DOC===`. Lines before the first separator are a file header of `#` comments.

```
# tdes-corpus v1

===DOC===
doc_id: web-0001
lang: en
script: Latn
title: Monsoon onset over the Western Ghats
---
Body text starts after the --- line and runs until the next ===DOC===.
```

Required keys are `doc_id`, `lang`, `script`. Optional keys: `title`,
`stage_hint` (`early` | `mid` | `anneal`), `reserved` (`true` locks the document
out of training until the anneal stage), `min_context` (an integer; the document
is only packed when the stage sequence length is at least this long).

## Role markers — agentic and reasoning lanes

In those two lanes the body is split into spans by role markers at the start of
a line. This is what makes the loss mask meaningful: the prompt and the tool
output are context, the model's own turns are graded.

| Marker | Role | Loss |
|---|---|---|
| `@user:` | user turn | context |
| `@think:` | model planning | graded |
| `@tool_call:` | model tool call | graded |
| `@tool_result:` | environment observation | context |
| `@answer:` | model final answer | graded |

A marker's content is the rest of that line plus every following line up to the
next marker.

## Why the content looks the way it does

The text is hand-authored rather than sampled from a public dataset for three
reasons: the repo stays small, the run needs no network, and the eval/test
overlap the firewall has to catch is a real overlap that can be pointed at in
the source rather than an injected fake.
