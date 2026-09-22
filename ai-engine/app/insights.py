"""
LLM-generated, plain-English insights about the winning model.
Always returns a plain string (never a list/None) so the frontend never has
to guess about the type. Falls back to a templated summary if no API key is
configured or the LLM call fails for any reason (network, quota, etc.) --
training must never fail just because the LLM step failed.
"""
import os


def generate_insights(top_features, dataset_info, task_type, target_col, retrieved_context=None):
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return _fallback_insights(top_features, target_col, reason="no OPENROUTER_API_KEY configured")

    try:
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"),
            timeout=20,
        )

        feature_text = ", ".join(f"{f} ({round(v, 3)})" for f, v in top_features) or "none available"
        context_text = "; ".join(retrieved_context) if retrieved_context else "none"

        prompt = f"""You are a data scientist. A {task_type} model was trained to predict "{target_col}".

Dataset info: {dataset_info}
Top important features: {feature_text}
Relevant past experiments (memory): {context_text}

In 3-5 concise bullet points, explain in simple terms:
- why these features likely matter for predicting {target_col}
- what patterns this suggests
Keep the whole answer under 150 words. Do not repeat the raw numbers back verbatim."""

        response = llm.invoke([HumanMessage(content=prompt)])
        text = (response.content or "").strip()
        return text if text else _fallback_insights(top_features, target_col, reason="LLM returned empty response")

    except Exception as e:
        print("⚠️ LLM insight generation failed:", e)
        return _fallback_insights(top_features, target_col, reason=str(e))


def _fallback_insights(top_features, target_col, reason=""):
    if not top_features:
        return f"Could not generate feature-based insights for '{target_col}' ({reason})."
    lines = [f"Automated summary for predicting '{target_col}' (LLM insights unavailable: {reason}):"]
    for f, v in top_features:
        lines.append(f"- '{f}' was one of the strongest signals (importance {round(v, 3)}).")
    return "\n".join(lines)
