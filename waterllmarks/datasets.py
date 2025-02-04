"""WaterLLMarks datasets."""

import abc
import uuid

from datasets import concatenate_datasets, load_dataset
from langchain_core.documents import Document


class WLLMKDataset(abc.ABC):
    """A Dataset for RAG experiments in the WaterLLMarks project.

    The Dataset class is an abstract class that defines the interface for a dataset used
    in the RAG experiments.

    Attributes
    ----------
    uri : str
        HuggingFace URL.
    corpus : list[Document]
        Get the corpus of the dataset as a list of Langchain's `Document`s. A corpus has
        the following fields: `id`, `page_content`, and `metadata`.
    qas : list[dict]
        Get the questions and answers of the dataset as a list of dictionaries with the
        following schema:
        ```
        {
            "id": str,
            "user_input": str,
            "reference": str,
            "reference_contexts": list[str],
            "reference_context_ids": list[str],
        }
    """

    def __init__(self):
        raise NotImplementedError


class LLMPaperDataset(WLLMKDataset):
    """AutoRAG's 2024 LLM Papers v1 dataset."""

    uri = "MarkrAI/AutoRAG-evaluation-2024-LLM-paper-v1"

    def __init__(self):
        def get_corpus():
            corpus = load_dataset(self.uri, name="corpus", split="train")
            docs = []
            for entry in corpus:
                metadata = {
                    k: (v if v is not None else "")
                    for k, v in entry["metadata"].items()
                }
                title, _, content = entry["contents"].partition("\n")
                if len(content) == 0:
                    content = title
                    metadata["title"] = ""
                else:
                    metadata["title"] = title.partition(" ")[2]
                doc = Document(
                    id=entry["doc_id"], page_content=content.strip(), metadata=metadata
                )
                docs.append(doc)
            return docs

        def get_qas():
            qas = load_dataset(self.uri, name="qa", split="train")
            return [
                {
                    "id": qa["qid"],
                    "user_input": qa["query"],
                    # For some reason, the dataset's `generation_gt` and `retrieval_gt`
                    # provided as lists. `retrieval_gt` in particular is a nested list
                    # with a single element.
                    "reference": qa["generation_gt"][0],
                    "reference_context_ids": qa["retrieval_gt"][0],
                    "reference_contexts": [
                        doc.page_content
                        for doc in self.corpus
                        if doc.id in qa["retrieval_gt"][0]
                    ],
                }
                for qa in qas
            ]

        self.corpus = get_corpus()
        self.qas = get_qas()


class RAGBenchDataset(WLLMKDataset):
    """RAG-12000 dataset."""

    uri = "rungalileo/ragbench"

    def __init__(self):
        dss = []
        for name in [
            "covidqa",
            "cuad",
            "delucionqa",
            "emanual",
            "expertqa",
            "finqa",
            "hagrid",
            "hotpotqa",
            "msmarco",
            "pubmedqa",
            "tatqa",
            "techqa",
        ]:
            dss.append(load_dataset(self.uri, name=name, split="train+validation+test"))

        ds = concatenate_datasets(dss)

        def get_corpus():
            corpus = []
            for entry in ds:
                for doc in entry["documents"]:
                    id = uuid.uuid4()
                    new_doc = Document(id=id, page_content=doc)
                    corpus.append(new_doc)

            return corpus

        self.corpus = get_corpus()

        def get_qas():
            qas = []
            for entry in ds:
                qa = {
                    "id": entry["id"],
                    "user_input": entry["question"],
                    "reference": entry["response"],
                }

                used_context_idx = set(
                    [int(key[0]) for key in entry["all_relevant_sentence_keys"]]
                )
                used_contexts = [entry["documents"][idx] for idx in used_context_idx]
                used_context_ids = [
                    doc.id for doc in self.corpus if doc.page_content in used_contexts
                ]
                qa["context_refs"] = used_context_ids

                qas.append(qa)

            return qas

        self.qas = get_qas()
