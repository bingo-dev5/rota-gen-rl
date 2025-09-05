from datasets import load_dataset
import random
from pydantic import BaseModel
from typing import List, Dict, Tuple
import os


class Question(BaseModel):
    q: str
    a: str


class Document(BaseModel):
    document_text: str
    questions: List[Question]


def load_documents() -> Tuple[List[Document], List[Document]]:
    ds = load_dataset("ServiceNow/repliqa")
    documents: Dict[str, Document] = {}

    for data in ds["repliqa_0"]:
        if data["document_id"] not in documents:
            documents[data["document_id"]] = Document(
                document_text=data["document_extracted"],
                questions=[],
            )
        documents[data["document_id"]].questions.append(
            Question(q=data["question"], a=data["answer"])
        )

    all_documents: List[Document] = []
    for doc_id in documents:
        all_documents.append(documents[doc_id])

    random.seed(80)
    random.shuffle(all_documents)

    val_size = int(os.getenv("VAL_SIZE", 91))
    train_size = int(os.getenv("TRAIN_SIZE", 3500))

    if train_size + val_size > len(all_documents):
        raise ValueError(
            f"Train size + val size ({train_size + val_size}) is greater than the total number of documents ({len(all_documents)})"
        )

    val_documents = all_documents[:val_size]
    train_documents = all_documents[val_size : val_size + train_size]

    print(f"Loaded {len(all_documents)} documents")
    print(f"Train set size: {len(train_documents)}")
    print(f"Val set size: {len(val_documents)}")

    return val_documents, train_documents
