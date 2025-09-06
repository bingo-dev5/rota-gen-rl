import art
import openai
import random
import json
import re
from pydantic import BaseModel
import time
import os

from rota_generator.get_judge_completion import (
    get_judge_completion,
)
from rota_generator.load_documents import Document

from openpipe.client import OpenPipe


op_client = OpenPipe()


class RotaScenario(BaseModel):
    doc: Document
    step: int = 0
    use_full: bool = False


@art.retry(exceptions=(openai.LengthFinishReasonError,))
async def rollout(
    model: art.Model,
    judge_model,
    scenario: RotaScenario,
    system_prompt,
) -> art.Trajectory:
    client = model.openai_client()

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": system_prompt,
            }
        ],
        reward=0,
        metrics={
            "coverage_score": 0,
            "preferences_score": 0,
            "fairness_score": 0,
            "compliance_score": 0,
            "overall_score": 0,
        },
    )

    rota_prompt = f"""You are a specialized AI assistant that generates rotas for employees based on their staff grade, schedules and preferences.

Here is a set of employee attributes: {scenario.doc.document_text}

Generate a rota that optimally assigns shifts to employees while adhering to the following constraints:
- Ensure that all shifts are covered adequately based on the required staffing levels.
- Respect employee preferences and availability as much as possible.
- Balance the workload fairly among all employees.
- Comply with labor regulations regarding maximum working hours and mandatory breaks.
Provide the rota in a clear and organized format."""

    trajectory.messages_and_choices.append(
        {"role": "user", "content": rota_prompt}
    )

    requested_at = int(time.time() * 1000)

    messages = trajectory.messages()
    completion = await client.chat.completions.create(
        model=model.inference_model_name,
        messages=messages,
        max_tokens=1000,
    )
    choice = completion.choices[0]
    if scenario.use_full:
        choice.message.content = scenario.doc.document_text
    trajectory.messages_and_choices.append(choice)
    summary = choice.message.content

    # -------------------------------------------------------------
    # Criteria-based self-judging (no gold answers available)
    # We ask a judge model to rate the rota against key constraints.
    # -------------------------------------------------------------
    judge_instructions = {
        "coverage": "All required shifts are filled with appropriate roles/grades; no gaps.",
        "preferences": "Respects stated availability & preferences; minimal conflicts or unwanted assignments.",
        "fairness": "Even / proportionate distribution of total hours and undesirable shifts across staff.",
        "compliance": "Adheres to labor limits (max weekly hours, daily caps, rest periods, breaks).",
    }

    judge_prompt = f"""
You are an expert rota auditor.

Evaluate the following generated rota against the staff attributes & constraints.

=== STAFF ATTRIBUTES & REQUIREMENTS ===\n{scenario.doc.document_text}\n
=== GENERATED ROTA ===\n{summary}\n
For each criterion assign a score between 0.0 and 1.0 (float) where:
0.0 = completely fails, 0.5 = partially adequate, 1.0 = fully satisfies.

Criteria definitions:
Coverage: {judge_instructions['coverage']}
Preferences: {judge_instructions['preferences']}
Fairness: {judge_instructions['fairness']}
Compliance: {judge_instructions['compliance']}

Also produce an overall score (weighted average: Coverage 0.35, Preferences 0.2, Fairness 0.2, Compliance 0.25).

Return ONLY a minified JSON object with keys: coverage, preferences, fairness, compliance, overall, reasoning.
reasoning = short textual justification (<= 60 words). No backticks, no extra commentary.
If data is insufficient to judge a criterion, estimate conservatively.
""".strip()

    raw_judge = await get_judge_completion(
        judge_prompt,
        judge_model,
        temperature=0.0,
        max_tokens=400,
    )

    def parse_scores(txt: str):
        # Try direct JSON parse, else extract first JSON object substring
        try:
            return json.loads(txt)
        except Exception:
            match = re.search(r"\{.*\}", txt, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    return {}
            return {}

    scores = parse_scores(raw_judge)

    coverage = float(scores.get("coverage", 0.0))
    preferences = float(scores.get("preferences", 0.0))
    fairness = float(scores.get("fairness", 0.0))
    compliance = float(scores.get("compliance", 0.0))
    overall = scores.get("overall")
    try:
        overall = float(overall)
    except Exception:
        # Recompute if judge failed to supply numeric overall
        overall = (
            0.35 * coverage
            + 0.20 * preferences
            + 0.20 * fairness
            + 0.25 * compliance
        )

    trajectory.metrics.update(
        {
            "coverage_score": coverage,
            "preferences_score": preferences,
            "fairness_score": fairness,
            "compliance_score": compliance,
            "overall_score": overall,
        }
    )

    trajectory.metrics["word_count"] = len(summary.split())
    trajectory.metrics["len"] = len(summary)
    trajectory.reward = overall

    # Optional: sporadically print judge reasoning for inspection
    if random.random() < 0.05:
        print("[Judge Evaluation]", raw_judge)

    if os.getenv("OPENPIPE_API_KEY"):
        try:
            op_client.report(
                requested_at=requested_at,
                received_at=int(time.time() * 1000),
                req_payload={
                    "model": model.name,
                    "messages": messages,
                    "metadata": {
                        "project": "summarize",
                        "step": scenario.step,
                        "percent": trajectory.metrics["percent"],
                        "percent_full": trajectory.metrics["percent_full"],
                        "percent_diff": trajectory.metrics["percent_diff"],
                        "word_count": trajectory.metrics["word_count"],
                        "len": trajectory.metrics["len"],
                    },
                },
                resp_payload=completion,
                status_code=200,
            )
        except Exception as e:
            print(f"Error reporting to OpenPipe: {e}")

    return trajectory
