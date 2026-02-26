"""Three ways to connect an LLM to your data, all powered by mloda.

Each approach is a DataCreator FeatureGroup that calls mloda internally
to fetch employee data, then differs in how it invokes the LLM.

Approach 1 (loop):  Anthropic API + tool-calling loop. Your code
                    dispatches each discover_features / run_features
                    tool call in a while-loop.
Approach 2 (bash):  1 x claude -p --allowedTools Bash. Claude imports
                    mloda and runs it in Python via Bash.
Approach 3 (ptc):   Programmatic Tool Calling. Claude writes Python
                    code in a code_execution sandbox that calls
                    discover_features / run_features as async functions.
                    The sandbox pauses, our code fulfills each tool call,
                    and the sandbox resumes.

Usage:
    python demo.py                          # run all 3 (ptc/loop need ANTHROPIC_API_KEY)
    python demo.py loop                     # run only loop approach
    python demo.py bash                     # run only bash approach
    python demo.py ptc                      # run only ptc approach (needs ANTHROPIC_API_KEY)
"""

import json
import subprocess
import sys
from typing import Any, Optional

import anthropic
import pandas as pd
from mloda.core.api.plugin_docs import get_feature_group_docs
from mloda.provider import BaseInputData, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, PluginLoader
from mloda.user import mloda as mlodaAPI

import ptc_mloda_demo.feature_groups.sample_data.sample_data_features  # noqa: F401
from ptc_mloda_demo.extenders.observability.observability_extender import ObservabilityExtender

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"

QUESTIONS = [
    "(a) Who are the top 3 highest-paid employees? (employee_id, department, salary)",
    "(b) What is the average salary per department?",
    "(c) Which employees have a performance_score > 90? (employee_id, department, performance_score)",
]

EMPLOYEE_FEATURES = [
    "employee_id",
    "department",
    "salary",
    "years_experience",
    "performance_score",
]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _claude_p(prompt: str, allowed_tools: str = "") -> str:
    """Run a single claude -p call and return the text result."""
    cmd = ["claude", "-p", "--output-format", "json"]
    if allowed_tools:
        cmd.extend(["--allowedTools", allowed_tools])
    result = subprocess.run(cmd, input=prompt, capture_output=True, text=True)
    if result.returncode != 0:
        return f"[claude -p failed (exit {result.returncode})]: {result.stderr}"
    return json.loads(result.stdout).get("result", result.stdout)


# ---------------------------------------------------------------------------
# LoopApproach: Anthropic API + tool-calling loop
# ---------------------------------------------------------------------------

LOOP_TOOLS = [
    {
        "name": "discover_features",
        "description": (
            "Discover available mloda feature groups and their supported feature names. "
            "Returns documentation for all loaded feature groups."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Optional filter by feature group name (partial match).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "run_features",
        "description": (
            "Run mloda to fetch data for the given feature names. Returns a CSV string of the resulting DataFrame."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of feature names to fetch via mloda.",
                },
            },
            "required": ["feature_names"],
        },
    },
]

LOOP_PROMPT = (
    "You are a data analyst with access to mloda, a plugin-based data framework.\n\n"
    "First, use discover_features to see what data is available.\n"
    "Then, use run_features to fetch the employee dataset.\n"
    "Finally, analyze the data and answer these questions:\n" + "\n".join(QUESTIONS)
)


def _handle_tool_call(name: str, inputs: dict) -> str:  # type: ignore[type-arg]
    """Dispatch a tool call (shared by LoopApproach and PtcApproach)."""
    if name == "discover_features":
        docs = get_feature_group_docs(name=inputs.get("name"))
        return json.dumps(
            [{"name": d.name, "features": list(d.supported_feature_names), "description": d.description} for d in docs]
        )
    if name == "run_features":
        feature_names = inputs["feature_names"]
        features = [Feature.not_typed(f) for f in feature_names]
        results = mlodaAPI.run_all(
            features, compute_frameworks=["PandasDataFrame"], function_extender={ObservabilityExtender()}
        )
        return results[0].to_csv(index=False)
    return json.dumps({"error": f"Unknown tool: {name}"})


class LoopApproach(FeatureGroup):
    """Anthropic API + tool-calling loop. Claude calls discover_features / run_features tools."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        client = anthropic.Anthropic()
        messages: list = [{"role": "user", "content": LOOP_PROMPT}]  # type: ignore[type-arg]

        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                tools=LOOP_TOOLS,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                break

            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    result = _handle_tool_call(block.name, block.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        texts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return pd.DataFrame({cls.get_class_name(): ["\n".join(texts)]})


# ---------------------------------------------------------------------------
# BashApproach: 1 x claude -p with Bash tool
# ---------------------------------------------------------------------------

BASH_PROMPT = (
    "You are a data analyst. You have access to a Python project with mloda installed.\n"
    "The project virtualenv is at /home/tom/project/demo/ptc-mloda-demo/.venv/\n\n"
    "Use Bash to run Python code that:\n"
    "1. Activates the virtualenv\n"
    "2. Imports mloda and fetches the employee dataset:\n"
    "   from mloda.user import Feature, PluginLoader, mloda as mlodaAPI\n"
    "   import ptc_mloda_demo.feature_groups.sample_data.sample_data_features\n"
    "   PluginLoader.all()\n"
    "   features = [Feature.not_typed(f) for f in " + repr(EMPLOYEE_FEATURES) + "]\n"
    "   results = mlodaAPI.run_all(features, compute_frameworks=['PandasDataFrame'])\n"
    "   print(results[0].to_string())\n\n"
    "3. Analyze the output and answer:\n" + "\n".join(QUESTIONS)
)


class BashApproach(FeatureGroup):
    """1 x claude -p call with Bash tool. Claude imports mloda and runs it in Python."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({cls.get_class_name(): [_claude_p(BASH_PROMPT, allowed_tools="Bash")]})


# ---------------------------------------------------------------------------
# PtcApproach: Programmatic Tool Calling (code_execution + tools)
# ---------------------------------------------------------------------------

PTC_TOOLS: list[dict[str, Any]] = [
    {"type": "code_execution_20260120", "name": "code_execution"},
    {
        "name": "discover_features",
        "description": (
            "Discover available mloda feature groups and their supported feature names. "
            "Returns documentation for all loaded feature groups."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Optional filter by feature group name (partial match).",
                },
            },
            "required": [],
        },
        "allowed_callers": ["code_execution_20260120"],
    },
    {
        "name": "run_features",
        "description": (
            "Run mloda to fetch data for the given feature names. Returns a CSV string of the resulting DataFrame."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of feature names to fetch via mloda.",
                },
            },
            "required": ["feature_names"],
        },
        "allowed_callers": ["code_execution_20260120"],
    },
]

PTC_PROMPT = (
    "You are a data analyst with access to mloda, a plugin-based data framework.\n\n"
    "You have two async functions available in your code sandbox:\n"
    "  - discover_features(name=None): discover available feature groups and their features\n"
    "  - run_features(feature_names=[...]): fetch data for the given feature names, returns CSV\n\n"
    "Write Python code that:\n"
    "1. Calls discover_features() to see what data is available\n"
    "2. Calls run_features() with the relevant feature names to fetch the employee dataset\n"
    "3. Parses the CSV and analyzes the data to answer these questions:\n" + "\n".join(QUESTIONS)
)


class PtcApproach(FeatureGroup):
    """Programmatic Tool Calling. Claude writes code that calls tools from inside the sandbox."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator({cls.get_class_name()})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        client = anthropic.Anthropic()
        container_id: Optional[str] = None
        messages: list = [{"role": "user", "content": PTC_PROMPT}]  # type: ignore[type-arg]

        while True:
            kwargs: dict[str, Any] = {
                "model": MODEL,
                "max_tokens": 16384,
                "tools": PTC_TOOLS,
                "messages": messages,
            }
            if container_id:
                kwargs["container"] = container_id
            response = client.messages.create(**kwargs)

            container_obj = getattr(response, "container", None)
            if container_obj:
                container_id = container_obj.id

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                break

            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    result = _handle_tool_call(block.name, block.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        texts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return pd.DataFrame({cls.get_class_name(): ["\n".join(texts)]})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

APPROACH_MAP = {"loop": "LoopApproach", "bash": "BashApproach", "ptc": "PtcApproach"}

if __name__ == "__main__":
    PluginLoader.all()

    if len(sys.argv) > 1:
        names = [APPROACH_MAP[sys.argv[1]]]
    else:
        names = list(APPROACH_MAP.values())

    results = mlodaAPI.run_all(
        names, compute_frameworks=["PandasDataFrame"], function_extender={ObservabilityExtender()}
    )

    for r in results:
        for col in [c for c in r.columns if c in APPROACH_MAP.values()]:
            print(f"\n{'=' * 60}")
            print(f"  {col}")
            print(f"{'=' * 60}")
            print(r[col].iloc[0])
