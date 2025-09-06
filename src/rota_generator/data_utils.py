import json
import re
import asyncio
from pathlib import Path
from typing import List, Set

from litellm import acompletion

INPUT_GENERATION_MODEL = "openrouter/moonshotai/kimi-k2"

# Prompts copied verbatim from original code (DO NOT MODIFY)


def parse_raw_generation(text: str) -> List[str]:
    text = text.strip()
    # Direct JSON
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        if isinstance(data, dict):
            for k in ("inputs", "examples", "data"):
                if k in data and isinstance(data[k], list):
                    return [str(x).strip() for x in data[k] if str(x).strip()]
    except Exception:
        pass
    # JSON array substring
    match = re.search(r"\[[\s\S]*?\]", text)
    if match:
        try:
            arr = json.loads(match.group(0))
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    # Bullet/numbered lines
    items: List[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if len(line) > 5:
            items.append(line)
    return items


async def generate_training_inputs(
    task_description: str,
    target_count: int,
    batch_size: int = 40,
    max_batches: int | None = None,
    save_path: Path | None = None,
) -> List[str]:
    collected: List[str] = []
    seen: Set[str] = set()
    batch_index = 0
    max_batches = max_batches or (target_count // batch_size + 5)
    print(
        f"Generating up to {target_count} training inputs in batches of {batch_size} (model={INPUT_GENERATION_MODEL})"
    )
    while len(collected) < target_count and batch_index < max_batches:
        remaining = target_count - len(collected)
        current_batch = min(batch_size, remaining)
        batch_index += 1
        system_prompt = f"""You are a helpful assistant that generates diverse, high-quality training inputs.

Task: {task_description}

Generate {current_batch} diverse INPUT examples that someone might provide for this task.
Make sure the inputs:
1. Cover a wide range of cases and edge cases
2. Are realistic and practical
3. Vary in length and complexity
4. Represent real-world scenarios

Only generate the INPUTS, not the outputs. RULER will evaluate the model's attempts automatically.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Generate {current_batch} input examples for the task described above. Return them in the form of a list.",
            },
        ]
        print(f"Generating {current_batch} training inputs...")
        attempt = 0
        max_attempts = 4
        added = 0
        while attempt < max_attempts and added == 0:
            attempt += 1
            try:
                response = await acompletion(
                    model=INPUT_GENERATION_MODEL,
                    messages=messages,
                    temperature=0.9,
                )
                raw = response.choices[0].message.content
                candidates = parse_raw_generation(raw)
                for c in candidates:
                    if c not in seen:
                        seen.add(c)
                        collected.append(c)
                        added += 1
                        if len(collected) >= target_count:
                            break
                print(
                    f"Batch {batch_index} attempt {attempt}: candidates={len(candidates)} added={added} total={len(collected)}"
                )
                if added == 0:
                    await asyncio.sleep(1.5)
            except Exception as e:
                print(
                    f"Batch {batch_index} attempt {attempt} exception: {e.__class__.__name__}: {e}"
                )
                await asyncio.sleep(2 + attempt)
        if added and save_path:
            try:
                save_path.write_text(
                    json.dumps(collected, indent=2, ensure_ascii=False)
                )
            except Exception as e:
                print(f"Warning: failed to save partial inputs: {e}")
    if len(collected) < target_count * 0.6:
        raise RuntimeError(
            f"Insufficient inputs generated: {len(collected)}/{target_count}."
        )
    return collected[:target_count]


def load_cached_training_inputs(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"Failed to load cached training inputs ({e}); returning empty list.")
        return []


def save_training_inputs(path: Path, inputs: List[str]) -> None:
    try:
        path.write_text(json.dumps(inputs, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Failed to write training inputs: {e}")


async def top_up_training_inputs(
    existing_inputs: List[str],
    target: int,
    task_description: str,
    path: Path,
    per_batch_cap: int = 60,
    max_attempts: int = 5,
) -> List[str]:
    if len(existing_inputs) >= target:
        return existing_inputs
    print(
        f"Top-up needed: have {len(existing_inputs)}, need {target}. Generating {target - len(existing_inputs)} more..."
    )
    existing = set(existing_inputs)
    attempts = 0
    while len(existing_inputs) < target and attempts < max_attempts:
        attempts += 1
        still_needed = target - len(existing_inputs)
        batch_target = min(per_batch_cap, still_needed)
        try:
            new_samples = await generate_training_inputs(
                task_description,
                target_count=batch_target,
                batch_size=min(40, batch_target),
                max_batches=None,
                save_path=None,
            )
        except Exception as e:
            print(f"Top-up attempt {attempts} failed: {e}")
            break
        added = 0
        for s in new_samples:
            if s not in existing:
                existing.add(s)
                existing_inputs.append(s)
                added += 1
                if len(existing_inputs) >= target:
                    break
        print(
            f"Top-up attempt {attempts}: added {added}, total {len(existing_inputs)}"
        )
        save_training_inputs(path, existing_inputs)
        if added == 0:
            break
    if len(existing_inputs) >= target:
        print(
            f"Top-up complete: reached {len(existing_inputs)} examples (target {target})."
        )
    else:
        print(
            f"Top-up ended with {len(existing_inputs)} examples (< target {target}). Proceeding."
        )
    return existing_inputs
