import art
import asyncio
from dotenv import load_dotenv
import random
from art.skypilot import SkyPilotBackend

from rota_generator import rollout, RotaScenario
from summarizer.load_documents import load_documents

load_dotenv()

AGENT_NAME = "agent-007"
PROJECT_NAME = "rota-generator"
CLUSTER_NAME = "rotagen-art"


async def main():
    val_documents, train_documents = load_documents()

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
                                RotaScenario(
                                    doc=document, step=current_step
                                ),
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
                            rollout(model, SummarizerScenario(doc=document))
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
