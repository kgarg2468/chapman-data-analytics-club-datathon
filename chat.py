import json
import traceback

import pandas as pd
import streamlit as st
from openai import OpenAI

from utils import get_data_summary

SYSTEM_PROMPT_TEMPLATE = """You are an expert data analyst assistant for an Energy Drink Orders dashboard.
You answer questions about a dataset of online energy drink orders.

{data_summary}

RULES:
- When the user asks a question that requires computation (averages, counts, correlations, filtering, top-N, etc.),
  respond with a JSON block containing the Python/pandas code to run, like this:
  ```json
  {{"code": "result = df.groupby('Gender')['Revenue'].mean()"}}
  ```
  The code will be executed against a pandas DataFrame called `df` (the filtered dataset).
  The variable `result` will be displayed to the user, so assign your final answer to `result`.
  Available libraries: pandas (as pd), numpy (as np).

- When the question is conversational or interpretive (explain a trend, suggest a strategy, summarize findings),
  answer directly in natural language. You may reference data facts from the summary above.

- Be concise and specific. Use numbers when available.
- If a question is ambiguous, ask for clarification.
- Format currency as AUD (e.g., $4.50).
- When showing tables or lists, use markdown formatting.
"""


def build_system_prompt(df: pd.DataFrame) -> str:
    summary = get_data_summary(df)
    return SYSTEM_PROMPT_TEMPLATE.format(data_summary=summary)


def execute_code(code: str, df: pd.DataFrame) -> str:
    """Safely execute pandas code against the dataset and return the result."""
    import numpy as np  # noqa: F811

    local_vars = {"df": df.copy(), "pd": pd, "np": np}
    try:
        exec(code, {"__builtins__": {}}, local_vars)
    except Exception:
        return f"Code execution error:\n```\n{traceback.format_exc()}\n```"

    result = local_vars.get("result")
    if result is None:
        return "Code ran but no `result` variable was set."

    if isinstance(result, pd.DataFrame):
        return result.to_markdown(index=False)
    if isinstance(result, pd.Series):
        return result.to_markdown()
    return str(result)


def extract_code_block(text: str) -> str | None:
    """Extract code from a ```json code block if present."""
    if "```json" not in text:
        return None
    try:
        json_str = text.split("```json")[1].split("```")[0].strip()
        data = json.loads(json_str)
        return data.get("code")
    except (json.JSONDecodeError, IndexError):
        return None


def get_chat_response(
    messages: list[dict],
    df: pd.DataFrame,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> str:
    """Send messages to OpenAI and return the response, executing code if needed."""
    client = OpenAI(api_key=api_key)

    system_msg = {"role": "system", "content": build_system_prompt(df)}
    full_messages = [system_msg] + messages

    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=0.3,
        max_tokens=2000,
    )

    assistant_text = response.choices[0].message.content or ""

    code = extract_code_block(assistant_text)
    if code:
        code_result = execute_code(code, df)
        text_before_code = assistant_text.split("```json")[0].strip()
        if text_before_code:
            return f"{text_before_code}\n\n{code_result}"
        return code_result

    return assistant_text
