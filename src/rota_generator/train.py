import art
import asyncio
from dotenv import load_dotenv
import random
import json
import os
from pathlib import Path
from typing import List
from art.skypilot import SkyPilotBackend
from art.rewards import ruler_score_group
from art.utils import iterate_dataset
from art.utils.litellm import convert_litellm_choice_to_openai

from litellm import acompletion
from pydantic import BaseModel, Field

from rota_generator import rollout, RotaScenario
from rota_generator.load_documents import load_documents


load_dotenv()

AGENT_NAME = "agent-007"
PROJECT_NAME = "rota-generator"
CLUSTER_NAME = "rotagen-art"
RULER_MODEL = (
    "openrouter/deepseek/deepseek-r1-0528"  # Model for RULER evaluation
)
SYSTEM_PROMPT_GENERATION_MODEL = "openrouter/moonshotai/kimi-k2"
INPUT_GENERATION_MODEL = "openrouter/moonshotai/kimi-k2"

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


class TrainingDataset(BaseModel):
    inputs: List[TrainingInput] = Field(description="List of training inputs")


async def generate_training_inputs(
    task_description: str, num_examples: int = 50
) -> List[str]:
    """Generate diverse training inputs for the given task"""

    system_prompt = f"""You are a helpful assistant that generates diverse, high-quality training inputs.

Task: {task_description}

Generate {num_examples} diverse INPUT examples that someone might provide for this task.
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
            "content": f"Generate {num_examples} input examples for the task described above. Return them in the form of a list.",
        },
    ]

    print(f"Generating {num_examples} training inputs...")

    inputs = []

    i = 0
    while i < 5 and len(inputs) < num_examples:
        i += 1
        response = await acompletion(
            model=INPUT_GENERATION_MODEL,
            messages=messages,
            response_format=TrainingDataset,
            temperature=1.0,
        )

        dataset = TrainingDataset.model_validate_json(
            response.choices[0].message.content
        )
        inputs = [ex.input for ex in dataset.inputs]

    if len(inputs) < num_examples:
        raise ValueError(f"Failed to generate {num_examples} training inputs.")

    return inputs


async def main():
    # Load documents for training / validation
    val_documents, train_documents = load_documents()
    # Load or generate training inputs (cached to avoid regeneration every run)
    ROOT_DIR = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT_DIR / "data"
    DATA_DIR.mkdir(exist_ok=True)
    TRAINING_INPUTS_PATH = DATA_DIR / "training_inputs.json"

    force_regen = bool(os.getenv("FORCE_REGENERATE_TRAINING_INPUTS"))

    if TRAINING_INPUTS_PATH.exists() and not force_regen:
        try:
            training_inputs = json.loads(TRAINING_INPUTS_PATH.read_text())
            print(
                f"Loaded {len(training_inputs)} cached training inputs from {TRAINING_INPUTS_PATH} (set FORCE_REGENERATE_TRAINING_INPUTS=1 to regenerate)."
            )
        except Exception as e:
            print(
                f"Failed to load cached training inputs ({e}); regenerating..."
            )
            training_inputs = await generate_training_inputs(
                TASK_DESCRIPTION, num_examples=300
            )
            TRAINING_INPUTS_PATH.write_text(
                json.dumps(training_inputs, indent=2, ensure_ascii=False)
            )
            print(
                f"Saved {len(training_inputs)} training inputs to {TRAINING_INPUTS_PATH}"
            )
    else:
        if force_regen and TRAINING_INPUTS_PATH.exists():
            print(
                "FORCE_REGENERATE_TRAINING_INPUTS set; regenerating training inputs..."
            )
        elif not TRAINING_INPUTS_PATH.exists():
            print("No cached training inputs found; generating...")

        training_inputs = await generate_training_inputs(
            TASK_DESCRIPTION, num_examples=300
        )
        TRAINING_INPUTS_PATH.write_text(
            json.dumps(training_inputs, indent=2, ensure_ascii=False)
        )
        print(
            f"Saved {len(training_inputs)} training inputs to {TRAINING_INPUTS_PATH}"
        )

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

    batch_size = 10  # Process this many documents per batch
    num_epochs = 1  # Number of complete passes through the training data

    start_step = await model.get_step()
    max_steps = 1000

    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch + 1}/{num_epochs}")
        # Shuffle training data at the beginning of each epoch
        random.shuffle(train_documents)

        # Calculate how many batches we can process in this epoch
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
                            rollout(model, RotaScenario(doc=document))
                            for _ in range(10)
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
