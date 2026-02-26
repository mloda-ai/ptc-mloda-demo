[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/mloda-ai/mloda-plugin-template/blob/main/LICENSE)
[![mloda](https://img.shields.io/badge/built%20with-mloda-blue.svg)](https://github.com/mloda-ai/mloda)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

# PTC + mloda Demo

3 ways to connect an LLM to your data. Same question, same two tools, very different plumbing.

## The 3 Approaches

**Loop** (traditional agentic loop): Your code dispatches every tool call in a while-loop. Full control, high token cost.

**Bash** (Claude Code style): One `claude -p` call with Bash access. Claude writes Python and runs it directly. No tool definitions needed.

**PTC** (Programmatic Tool Calling): Claude writes Python in a sandboxed container, calls your tools as async functions. Results stay in the sandbox, only `print()` output goes back.

## The Code

### 1/3: The Agentic Loop

Every tool call is a round-trip. Every result lands in context.

```python
while True:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        tools=TOOLS,
        messages=messages,
    )
    if response.stop_reason == "end_turn":
        break

    for block in response.content:
        if block.type == "tool_use":
            result = handle_tool(block.name, block.input)
            #        ^^^^ full result goes into context window
```

### 2/3: The Bash Shortcut

No tool loop. Claude just writes code. Fast, but broad permissions.

```python
subprocess.run(
    ["claude", "-p", "--allowedTools", "Bash"],
    input="Use mloda to fetch employee data and find top earners",
)
# Claude writes + runs Python. No tool definitions needed.
# But: full shell access, limited observability.
```

### 3/3: Programmatic Tool Calling

Sandboxed. Tool results stay in the sandbox. Only print() goes back.

```python
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=16384,
    tools=[
        {"type": "code_execution_20260120", "name": "code_execution"},
        {
            "name": "run_features",
            "description": "Fetch data for given feature names. Returns CSV.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "feature_names": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["feature_names"],
            },
            "allowed_callers": ["code_execution_20260120"],  # callable from sandbox
        },
    ],
    messages=messages,
)

# Claude writes and runs this inside the sandbox:
#
#   csv = await run_features(feature_names=["employee_id", "salary", "department"])
#   df = pd.read_csv(io.StringIO(csv))
#   top3 = df.nlargest(3, "salary")
#   print(top3.to_string())   # only print() output goes back to Claude
```

## Quick Start

```bash
uv venv && source .venv/bin/activate
uv sync --all-extras
```

Run all 3 approaches:
```bash
python demo.py
```

Run a single approach:
```bash
python demo.py loop
python demo.py bash
python demo.py ptc
```

> `loop` and `ptc` require an `ANTHROPIC_API_KEY`. `bash` requires `claude` CLI installed.

## How It Works

All 3 approaches use the same two tools:

- `discover_features` -- list available mloda feature groups and their features
- `run_features` -- fetch data for given feature names via `mloda.run_all()`

The data comes from a hardcoded employee dataset (id, department, salary, experience, performance score). Each approach answers the same 3 questions about this data.

The difference is only in how the model reaches the tools.

## Project Structure

```
demo.py                           # all 3 approaches in one file
ptc_mloda_demo/
  feature_groups/sample_data/     # employee dataset (FeatureGroup)
  extenders/observability/        # observability extender
tests/
  test_mloda_imports.py
```

## Checks

```bash
tox
```

## Links

- [mloda](https://github.com/mloda-ai/mloda) -- core library
- [mloda.ai](https://mloda.ai) -- project website
- [PTC docs](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/programmatic-tool-calling) -- Anthropic's Programmatic Tool Calling
