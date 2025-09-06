import art
import asyncio
from dotenv import load_dotenv
import random
import json
import os
from pathlib import Path
from typing import List
from art.skypilot import SkyPilotBackend

from litellm import acompletion
from pydantic import BaseModel, Field

from rota_generator import rollout, RotaScenario
from rota_generator.load_documents import Document
from rota_generator.data_utils import (
    generate_training_inputs,
    top_up_training_inputs,
    load_cached_training_inputs,
    save_training_inputs,
)


load_dotenv()

AGENT_NAME = "agent-007"
PROJECT_NAME = "rota-generator"
CLUSTER_NAME = "rotagen-art"
# (Optional) A separate model could be used for judging; rollout handles judging internally now.
RULER_MODEL = "openrouter/deepseek/deepseek-r1-0528"
SYSTEM_PROMPT_GENERATION_MODEL = "openrouter/moonshotai/kimi-k2"
INPUT_GENERATION_MODEL = "openrouter/moonshotai/kimi-k2"
TARGET_TRAINING_INPUTS = 300  # Desired total number of training examples

TASK_DESCRIPTION = """
You are a specialized AI assistant that generates fully functioning rotas for employees based on their staff grade, schedules and preferences.
Each employee has the following attributes:
- Staff grade
- Preferred working hours
- Availability
- Skill set
Your task is to create a rota that optimally assigns shifts to employees while adhering to the following constraints:
- Ensure that all shifts are covered adequately based on the required staffing levels.
- Respect employee preferences and availability as much as possible.
- Balance the workload fairly among all employees.
- Comply with labor regulations regarding maximum working hours and mandatory breaks.
"""


class TrainingInput(BaseModel):
    input: str = Field(description="The input text for the task")


class TrainingDataset(BaseModel):  # kept for backward compatibility
    inputs: List[TrainingInput] = Field(description="List of training inputs")


# Generate a system prompt for the task
async def generate_system_prompt(
    task_description: str,
) -> str:
    """Generate an appropriate system prompt for the task"""

    messages = [
        {
            "role": "system",
            "content": "Generate a clear, concise system prompt for a model that will perform the following task. The prompt should be direct and instructional.",
        },
        {
            "role": "user",
            "content": f"Task: {task_description}\n\nGenerate a system prompt for this task.",
        },
    ]

    response = await acompletion(
        model=SYSTEM_PROMPT_GENERATION_MODEL,
        messages=messages,
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


async def main():
    # We synthesize training documents instead of loading gold data.
    # Load or generate training inputs (cached to avoid regeneration every run)
    ROOT_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True)
    TRAINING_INPUTS_PATH = DATA_DIR / "training_inputs.json"

    force_regen = bool(os.getenv("FORCE_REGENERATE_TRAINING_INPUTS"))

    if TRAINING_INPUTS_PATH.exists() and not force_regen:
        training_inputs = load_cached_training_inputs(TRAINING_INPUTS_PATH)
        print(
            f"Loaded {len(training_inputs)} cached training inputs from {TRAINING_INPUTS_PATH} (set FORCE_REGENERATE_TRAINING_INPUTS=1 to regenerate)."
        )
    else:
        training_inputs = []

    if force_regen or not training_inputs:
        if force_regen and TRAINING_INPUTS_PATH.exists():
            print(
                "FORCE_REGENERATE_TRAINING_INPUTS set; regenerating training inputs..."
            )
        elif not TRAINING_INPUTS_PATH.exists():
            print("No cached training inputs found; generating...")
        training_inputs = await generate_training_inputs(
            TASK_DESCRIPTION,
            target_count=TARGET_TRAINING_INPUTS,
            save_path=TRAINING_INPUTS_PATH,
        )
        save_training_inputs(TRAINING_INPUTS_PATH, training_inputs)
        print(
            f"Saved {len(training_inputs)} training inputs to {TRAINING_INPUTS_PATH}"
        )

    training_inputs = await top_up_training_inputs(
        training_inputs,
        TARGET_TRAINING_INPUTS,
        TASK_DESCRIPTION,
        TRAINING_INPUTS_PATH,
    )

    # Wrap into Document objects (empty questions list for compatibility)
    documents = [
        Document(document_text=t, questions=[]) for t in training_inputs
    ]
    random.shuffle(documents)
    val_split = max(10, len(documents) // 6)
    val_documents = documents[:val_split]
    train_documents = documents[val_split:]

    SYSTEM_PROMPT = await generate_system_prompt(TASK_DESCRIPTION)
    print(f"Generated system prompt:\n\n{SYSTEM_PROMPT}")

    backend = await SkyPilotBackend.initialize_cluster(
        cluster_name=CLUSTER_NAME,
        env_path=".env",
        gpu="L4",
    )

    model = art.TrainableModel(
        name=AGENT_NAME,
        project=PROJECT_NAME,
        base_model="Qwen/Qwen2.5-3B-Instruct",
    )
    await backend._experimental_pull_from_s3(model)
    await model.register(backend)

    batch_size = 10  # documents per batch
    num_epochs = 1
    start_step = await model.get_step()
    max_steps = 1000

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        random.shuffle(train_documents)
        num_batches = min(
            len(train_documents) // batch_size,
            (max_steps - start_step) // num_epochs,
        )
        for batch in range(num_batches):
            current_step = start_step + epoch * num_batches + batch
            if current_step >= max_steps:
                break
            print(
                f"Epoch {epoch + 1}, Batch {batch + 1}/{num_batches}, Step {current_step}"
            )
            batch_start_idx = batch * batch_size
            batch_end_idx = (batch + 1) * batch_size

            val_groups, train_groups = await asyncio.gather(
                art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(
                            rollout(
                                model,
                                RotaScenario(doc=document, step=current_step),
                                SYSTEM_PROMPT,
                            )
                            for _ in range(2)
                        )
                        for document in val_documents
                    ),
                    pbar_desc=f"gather val (epoch {epoch + 1})",
                ),
                art.gather_trajectory_groups(
                    (
                        art.TrajectoryGroup(
                            rollout(
                                model,
                                RotaScenario(doc=document),
                                SYSTEM_PROMPT,
                            )
                            for _ in range(2)
                        )
                        for document in train_documents[
                            batch_start_idx:batch_end_idx
                        ]
                    ),
                    pbar_desc=f"gather train (epoch {epoch + 1}, batch {batch + 1})",
                ),
            )

            await model.log(val_groups)
            await model.delete_checkpoints()
            await model.train(
                train_groups,
                config=art.TrainConfig(learning_rate=5e-5),
            )
            await backend._experimental_push_to_s3(model)


if __name__ == "__main__":
    asyncio.run(main())
