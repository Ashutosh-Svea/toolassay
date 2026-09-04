# toolassay

[![CI](https://github.com/Ashutosh-Svea/toolassay/actions/workflows/ci.yml/badge.svg)](https://github.com/Ashutosh-Svea/toolassay/actions/workflows/ci.yml)

An MCP server's tool contracts, meaning the names, descriptions, input schemas, result
shapes and error text it advertises, are the only thing a model has to go on when it
decides which tool to call, with what arguments, and whether the result answered the
question. When a description is vague, the model guesses. When a result leaves out the id
the next call needs, the model improvises: it reaches for a bigger tool, or retries blind.
Every detour is paid for in turns and tokens, and none of it shows up in the server's own
tests, because the server did exactly what it was asked.

toolassay measures that gap. Give it an MCP server and a file of natural-language cases.
It discovers the server's tools over the standard `tools/list` call, hands them to a model
as tool definitions, asks the model to complete each case, and records which tools it
picked, whether it finished, and what the run cost. Run it before and after a contract
change, then `toolassay diff` tells you whether the change made things cheaper without
making them worse, and exits non-zero when it did not. That makes a tool description
something you can change with a CI gate behind you.

Generic LLM evaluation harnesses exist. This one is specific to MCP: the unit under test is
the contract, the fixture is a live server on stdio or streamable HTTP, and the metrics are
the ones a contract change moves.

## Quickstart

Requires Python 3.11 or newer and an Anthropic API key in `ANTHROPIC_API_KEY`.

```bash
git clone https://github.com/Ashutosh-Svea/toolassay.git && cd toolassay
```

```bash
make install
```

```bash
source .venv/bin/activate
```

The bundled example is a bookshop MCP server with four tools and two versions of their
contracts. `v1` is how contracts often ship; `v2` fixes the contracts and changes nothing
else. Run the same cases against both, then diff:

```bash
toolassay run --server examples/bookshop/server-v1.yaml --cases examples/bookshop/cases.yaml --out before.run.json
```

```bash
toolassay run --server examples/bookshop/server-v2.yaml --cases examples/bookshop/cases.yaml --out after.run.json
```

```bash
toolassay diff before.run.json after.run.json --max-cost-increase 10%
```

`make demo` runs those three commands in sequence. The default model is `claude-opus-5`
at low effort; the whole demo costs well under a dollar.

## What a run records

For every case, the artifact holds:

| Field | Meaning |
| --- | --- |
| `tool_selection_correct` | The tools the model called matched `expected_tools` |
| `args_correct` | The first call to each tool in `expected_args` carried the expected values |
| `answer_correct` | The final answer contained every `expected_substrings` entry |
| `completed` | The model finished normally and the answer checks (and the judge, if on) passed |
| `passed` | `completed`, and no tool or argument check failed |
| `turns` | Model requests made for the case |
| `usage` | Input, output, and cache token counts, straight from the provider |
| `cost_usd` | Estimated from the token counts and a price table |
| `latency_ms` | Wall-clock time for the case, including tool execution |
| `tool_calls` | The full trace: turn, tool, arguments, result text, error flag, latency |
| `failures` | Plain-language reasons for anything that did not pass |

The artifact also stores the tool definitions the server advertised, so a diff can say
which contracts changed between two runs, and a redacted copy of the server config.

```json
{
  "id": "reserve-with-date",
  "passed": false,
  "completed": true,
  "tool_selection_correct": false,
  "args_correct": false,
  "turns": 3,
  "usage": {"input_tokens": 3984, "output_tokens": 239, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
  "cost_usd": 0.0259,
  "called_tools": ["reserve_book", "reserve_book"],
  "failures": [
    "expected tools ['reserve_book'] (exact), model called ['reserve_book', 'reserve_book']",
    "reserve_book: argument 'pickup_date' was '12 September 2026', expected '2026-09-12'"
  ]
}
```

## A worked example: fixing a contract and proving it

The bookshop server ([src/toolassay/demo/server.py](src/toolassay/demo/server.py)) has
four tools: `find_book`, `get_book`, `list_inventory`, and `reserve_book`. The `v1`
contracts have four common flaws:

- `find_book` is described as "Find a book." and returns titles and authors but no
  `book_id`, which is exactly what `get_book` and `reserve_book` need next.
- `list_inventory` takes no parameters and returns all 200 books, so any question it is
  used for costs several thousand input tokens on the following turn.
- `reserve_book` answers every bad input with "Error: invalid request", including a date
  written as "12 September 2026" when it wanted `2026-09-12`.
- Nothing tells the model which tool to use when it already has a `book_id`.

The `v2` contracts keep the same tool names and the same underlying behaviour. They
change the descriptions, add a `book_id` to `find_book` results, give `list_inventory` a
`section` filter and a `limit`, and make `reserve_book` say what it wanted:

```text
v1  find_book: Find a book.

v2  find_book: Search the catalogue by words from a title or an author's name
    (case-insensitive). Returns up to 5 matches, each with book_id, title, author,
    section, price and stock. Use get_book with a book_id when you also need the
    shelf location.
```

```text
v1  reserve_book: Reserve a book.

v2  reserve_book: Reserve a book for in-store pickup. book_id is the BK-0000 id,
    customer_email is the customer's email address, and pickup_date must be an ISO
    date (YYYY-MM-DD), for example 2026-09-12. Returns a reservation_id, or a reason
    the reservation could not be made.
```

The case file ([examples/bookshop/cases.yaml](examples/bookshop/cases.yaml)) is the same
for both. Here is the shape of what `toolassay diff` prints for the two runs. This table
was produced by running the three commands above with toolassay's offline scripted model
(the same test double the test suite uses) standing in for Claude, so the token counts
are illustrative; run `make demo` with a key to see real ones.

```text
                                  toolassay diff: before vs after
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ case              ┃ status ┃       turns ┃                 tokens ┃                         cost ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ find-by-author    │ pass   │ 2 to 2 (+0) │    2573 to 3112 (+539) │ $0.0149 to $0.0176 (+0.0027) │
│ price-by-id       │ pass   │ 2 to 2 (+0) │    2572 to 3034 (+462) │ $0.0146 to $0.0168 (+0.0022) │
│ shelf-location    │ fixed  │ 4 to 3 (-1) │ 22074 to 4904 (-17170) │ $0.1144 to $0.0274 (-0.0870) │
│ stock-count       │ pass   │ 3 to 2 (-1) │  12232 to 3109 (-9123) │ $0.0639 to $0.0174 (-0.0465) │
│ cheapest-poetry   │ pass   │ 2 to 2 (+0) │  10804 to 3248 (-7556) │ $0.0555 to $0.0182 (-0.0373) │
│ reserve-with-date │ fixed  │ 3 to 2 (-1) │   4223 to 3171 (-1052) │ $0.0259 to $0.0187 (-0.0072) │
│ unknown-title     │ pass   │ 2 to 2 (+0) │    2543 to 3034 (+491) │ $0.0146 to $0.0170 (+0.0024) │
└───────────────────┴────────┴─────────────┴────────────────────────┴──────────────────────────────┘
tool contract changes: find_book (description), get_book (description), list_inventory
(description), list_inventory (schema), reserve_book (description)
pass rate 71% to 100%, turns 18 to 15, tokens 57021 to 23612, cost $0.3037 to $0.1330 (-56.2%)
gates passed
```

Read it row by row. The longer `v2` descriptions cost about 500 extra input tokens on
every case, which is the price of being specific. The cases that used to go through the
200-row dump got cheaper by an order of magnitude. `shelf-location` went from a wrong
tool sequence to the expected one because `find_book` now returns the id. And
`reserve-with-date` stopped needing a retry because the description states the date
format. The run as a whole passes more and costs less, so the gate passes. Had the
change raised cost past the threshold or dropped the pass rate, `diff` would exit 1 and
say why.

## Writing cases

A case file is YAML with optional shared settings and a list of cases:

```yaml
system: You are the assistant for a small bookshop. Keep answers short.
max_turns: 8
cases:
  - id: shelf-location
    prompt: Where is "The Quiet Lantern" shelved?
    expected_tools: [find_book, get_book]
    expected_substrings: ["P3"]

  - id: reserve-with-date
    prompt: Reserve BK-0088 for pat@example.com, for pickup on 12 September 2026.
    expected_tools: reserve_book
    expected_args:
      reserve_book:
        book_id: BK-0088
        pickup_date: "2026-09-12"
    expected_substrings: ["RSV-"]

  - id: unknown-title
    prompt: Is "The Copper Almanac of Nowhere" in stock?
    expected_tools: [find_book]
    tool_match: contains
    judge: >-
      The answer states clearly that no such book was found and does not invent a
      title, price, or stock count.
```

Per case:

| Key | Purpose |
| --- | --- |
| `id` | Unique name; the diff matches cases across runs by it |
| `prompt` | What the user asks |
| `expected_tools` | A tool name or an ordered list. Omit to skip the check; `[]` means no tool should be called |
| `tool_match` | `exact` (default): the call sequence must equal the list. `contains`: the list must appear in order, extra calls allowed |
| `expected_args` | Per tool, argument values the first call to that tool must contain (a subset is fine) |
| `expected_substrings` | Strings the final answer must contain, matched case-insensitively |
| `judge` | A rubric for the LLM judge; only used when the run passes `--judge` |
| `system`, `max_turns` | Per-case overrides of the file-level values |
| `tags` | Free-form labels stored in the artifact |

Unknown keys are rejected, so a typo in `expected_tool` fails loudly instead of silently
skipping a check. `toolassay validate --cases cases.yaml` checks a file without running it.

## Pointing at a server

Server configs are small YAML (or JSON) files. Over stdio, toolassay launches the server
as a subprocess:

```yaml
transport: stdio
command: ${TOOLASSAY_PYTHON}
args: ["-m", "toolassay.demo.server", "--contracts", "v1"]
env:
  LOG_LEVEL: warning
cwd: .
```

Over streamable HTTP, it connects to a server that is already running:

```yaml
transport: http
url: http://127.0.0.1:8765/mcp
headers:
  Authorization: Bearer ${BOOKSHOP_TOKEN}
```

`${NAME}` expands from the environment; a missing variable is an error, so a config can
never silently send an empty token. `${TOOLASSAY_PYTHON}` is built in and expands to the
interpreter running toolassay. Put secrets behind `${NAME}` references: the artifact and
the spans carry the config as written, with the references intact, header and env values
masked, and URLs stripped of userinfo and query strings. A secret typed straight into the
file would be stored as typed.

`toolassay tools --server server.yaml` prints what the server advertises, which is a
quick way to read a contract the way the model will.

## How scoring works

Scoring is deterministic first. Tool selection, arguments, and substrings are plain
comparisons, so a run that uses only those checks is fully reproducible given the same
model responses, and costs nothing beyond the model calls the cases need.

- `completed` means the model stopped normally (no refusal, no `max_tokens` cut-off, no
  tool calls left pending at `max_turns`), every substring was present, and the judge, if
  it ran, passed.
- `passed` means `completed` plus no failed tool-selection or argument check. A model can
  complete a task with the wrong tool; the two flags keep those outcomes apart.
- Turns count model requests. A case that needs one tool call takes two turns: the request
  that produced the call, and the request that produced the answer.

The LLM judge is a separate, opt-in path for cases whose answers are free text. Pass
`--judge` (and optionally `--judge-model`) and every case with a `judge` rubric is graded
by a second conversation that sees the rubric, the prompt, the tool calls made, and the
final answer, and must reply with `{"pass": ..., "reason": ...}`. Judge token usage and
cost are recorded separately from the case's own, and an unparseable judge reply counts as
a fail rather than a pass. Runs without `--judge` never touch the judge code.

## The diff gate

`toolassay diff baseline.json candidate.json` compares the two runs case by case and
reports pass/fail transitions, turn deltas, token deltas, and cost deltas. It then checks
four gates:

- `--max-regressions` (default `0`): cases that passed in the baseline and fail in the
  candidate. This is per case, so a newly fixed case can never hide a newly broken one
  behind an unchanged pass rate.
- Missing cases: a baseline case absent from the candidate fails the gate unless you pass
  `--allow-missing-cases`, because a deleted failing case would otherwise look like an
  improvement.
- `--max-pass-rate-drop` (default `0`, in percentage points), over the cases both runs
  share.
- `--max-cost-increase` (default `0%`): a percentage of the baseline cost over the shared
  cases, or an absolute amount in USD such as `0.05`. `--no-cost-gate` skips it.

Any violation exits with status 1 and a sentence naming the gate and the numbers. Case ids
present in only one run are listed and left out of the totals. `--json` prints the report
as JSON for other tooling.

## Telemetry

Every run emits OpenTelemetry spans that follow the GenAI semantic conventions (still in
development status upstream, attribute names checked against
`opentelemetry-semantic-conventions` 0.65b0) plus the MCP conventions from the same
specification:

```text
toolassay.run                     one span per run
  invoke_agent toolassay          one per case, with gen_ai.conversation.id = run/case
    chat {model}                  one per model request: request and response model,
                                  finish reason, input and output token counts
    execute_tool {tool}           one per tool call: gen_ai.tool.name, gen_ai.tool.call.id,
                                  gen_ai.tool.type, mcp.method.name, network.transport,
                                  and error.type when the tool returned an error
```

The exporter is OTLP over HTTP by default and honours the usual
`OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` variables, or
`--otel-endpoint`. If nothing is listening at the resolved endpoint, the run prints one
note and carries on without exporting rather than retrying into a wall. `--otel console`
prints spans to stderr; `--otel none` disables them. Tool arguments and results are only
attached to spans with `--otel-content`, matching the conventions' opt-in rule for
content.

## Models and providers

The Anthropic adapter is the first one and the reference for the interface. It uses the
official `anthropic` SDK, defaults to `claude-opus-5`, sends every discovered tool as a
tool definition with `strict: true` and `additionalProperties: false` on every object in
the schema (falling back to non-strict for a tool whose schema already allows extra
properties), reads tool calls from `tool_use` content blocks, loops while `stop_reason`
is `tool_use`, and takes token counts from `response.usage`. Tool selection is not a hard
reasoning task, so the default `output_config` effort is `low`; `--effort` changes it.
`--no-strict` sends schemas exactly as the server published them.

Adding a provider means implementing two small classes from
[src/toolassay/adapters/base.py](src/toolassay/adapters/base.py): a `ModelAdapter` that
turns tool definitions into the provider's shape, and a `Conversation` with `send_user`
and `send_tool_results`, each returning a provider-neutral `ModelTurn`. Register it with
`toolassay.adapters.register("name", factory)` and select it with `--provider name`.
There is no lowest-common-denominator abstraction over every provider's request format;
each adapter owns its own history.

## Cost estimation

Costs come from a built-in USD-per-million-token table for current Claude models
(snapshot dated in [src/toolassay/pricing.py](src/toolassay/pricing.py)) with cache write
and cache read multipliers applied to cache token counts. A model with no known price
gets `cost_usd: null` rather than a wrong number, and the cost gate then reports that it
cannot be evaluated. Override or extend the table with `--prices prices.yaml`:

```yaml
claude-opus-5: {input: 5.0, output: 25.0}
my-fine-tune: {input: 1.5, output: 6.0}
```

## Design decisions

Where a choice was open, the simpler option won. The ones that shape results:

- **Exact tool matching by default.** Extra or repeated calls are a cost signal, and the
  diff is where they should show up. `tool_match: contains` is there for cases where the
  route does not matter.
- **Arguments are checked on the first call to a tool.** A retry that eventually gets it
  right still marks `args_correct` false, because the contract made the model get it
  wrong first. The `completed` flag still records that the task finished.
- **Strict schemas change one thing.** Only `additionalProperties: false` is added, so the
  model sees the contract as the server wrote it. A schema that already allows extra
  properties somewhere (a dict-typed parameter, say) cannot be closed without changing
  its meaning, so that tool is sent without `strict`, the run says so, and the artifact
  lists it under `relaxed_tools`. If the API rejects a schema feature outright, the run
  stops with a hint, and `--no-strict` is the escape hatch.
- **Every gate defaults to zero tolerance.** Any regression fails unless you say how much
  you will accept. `--max-cost-increase 10%` is the usual CI setting given run-to-run
  token noise.
- **The judge is opt-in and kept apart.** Deterministic runs stay reproducible; judge
  cost is reported on its own line.
- **Spans are richer than one per run plus one per tool call.** Per-case and per-request
  spans carry the token counts where the conventions put them, and every tool call is a
  child of its case.
- **Fatal provider errors stop the run.** Bad credentials or an unknown model id would
  fail every case identically, so they exit with status 2 instead of writing an artifact
  full of errors. Any other model or tool failure is recorded on its case and the run
  continues.
- **Name.** The obvious name, toolgauge, was already taken on GitHub by two projects in
  the same space, so this one is toolassay: an assay tests what something is made of and
  whether it is up to standard.

## Repository layout

```text
src/toolassay/
  cli.py            run, diff, tools, validate
  cases.py          case file schema and loader
  server.py         MCP connection over stdio, streamable HTTP, or in-process
  adapters/         ModelAdapter interface, registry, and the Anthropic adapter
  runner.py         the tool-use loop, one case at a time
  scoring.py        deterministic checks
  judge.py          optional LLM judge
  pricing.py        price table and cost estimate
  telemetry.py      GenAI semantic-convention spans
  artifact.py       run artifact model, read and write
  diff.py           comparison and gates
  report.py         terminal tables
  demo/             the bookshop server and its invented catalogue
examples/bookshop/  server configs (v1, v2, http) and the case file
tests/              offline suite with a scripted model adapter
```

## Development

```bash
make check
```

That runs ruff (lint and format), mypy in strict mode, and pytest. The tests never call a
model: a scripted adapter plays the model, and the demo server is exercised in-process,
over stdio as a subprocess, and over streamable HTTP on a local port. CI runs the same
three steps on Python 3.11, 3.12, and 3.13.

## License

MIT. See [LICENSE](LICENSE).
