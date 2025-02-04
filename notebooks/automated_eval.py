import argparse
import logging
from functools import partial

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import (
    ConfigurableField,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from waterllmarks.datasets import LLMPaperDataset
from waterllmarks.evaluation import DEFAULT_ALL_METRICS, WLLMKResult, evaluate
from waterllmarks.pipeline import DictParser, RunnableTryFix
from waterllmarks.watermarks import Rizzo2016, TokenWatermark

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def run_pipeline(seed):
    logging.info(f"Starting pipeline with seed: {seed}")

    # Setup LLM client and embeddings
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=7,
        check_every_n_seconds=0.1,
        max_bucket_size=10,
    )
    llm_client = ChatOpenAI(
        model="gpt-4o-mini",
        seed=seed,
        rate_limiter=rate_limiter,
    )
    embedding_openai = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        embedding_function=embedding_openai,
        persist_directory="./chroma_langchain_db",
    )

    # Load dataset
    ds = LLMPaperDataset()
    logging.info("Dataset loaded")

    # Define retriever
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"search_k": 20}
    ).configurable_alternatives(
        ConfigurableField(id="retriever"),
        default_key="chroma",
        empty=RunnableLambda(lambda _: []),
    )
    logging.info("Retriever defined")

    # Define prompt templates and chains
    prompt = PromptTemplate(
        input_variables=["pipeline_input"],
        template="Answer the question. Keep the answer short and concise.\n\nQuestion: {pipeline_input}\n\nAnswer:\n",
    )
    set_input = RunnablePassthrough.assign(pipeline_input=DictParser("user_input"))
    output_parser = StrOutputParser(name="content")
    llm = RunnablePassthrough.assign(response=prompt | llm_client | output_parser)
    norag_chain = set_input | llm

    rag_prompt = PromptTemplate(
        input_variables=["pipeline_input", "context"],
        template="Answer the question using the provided context. Keep the answer short and concise.\n\nContext:\n{context}\n\nQuestion: {pipeline_input}\n\nAnswer:\n",
    )
    context_formatter = RunnableLambda(
        lambda docs: "\n\n".join([doc.page_content for doc in docs])
    )
    ragllm = (
        RunnablePassthrough.assign(
            retrieved_contexts=DictParser("pipeline_input") | retriever
        )
        | RunnablePassthrough.assign(
            context=DictParser("retrieved_contexts") | context_formatter
        )
        | RunnablePassthrough.assign(response=rag_prompt | llm_client | output_parser)
    )
    rag_chain = set_input | ragllm
    logging.info("Prompt templates and chains defined")

    # Baseline results
    logging.info("Generating baseline results")
    baseline_results = rag_chain.batch(ds.qas)
    baseline_qas = [
        {
            "id": res["id"],
            "user_input": res["user_input"],
            "reference": res["response"],
            "reference_contexts": [
                doc.page_content for doc in res["retrieved_contexts"]
            ],
            "reference_context_ids": [doc.id for doc in res["retrieved_contexts"]],
        }
        for res in baseline_results
    ]
    logging.info("Baseline results generated")

    # TokenWatermark
    wtmk = TokenWatermark(key=b"0123456789ABCDEF")
    apply_watermark = RunnablePassthrough.assign(
        pipeline_input=DictParser("pipeline_input") | wtmk.apply_as_runnable()
    )
    token_rag_chain = set_input | apply_watermark | ragllm
    logging.info("TokenWatermark chain defined")

    # Rizzo2016 (Character Embedding)
    wtmk = Rizzo2016(key=b"0123456789ABCDEF")
    apply_watermark = RunnablePassthrough.assign(
        pipeline_input=DictParser("pipeline_input") | wtmk.apply_as_runnable()
    )
    augment_prompt = PromptTemplate(
        input_variables=["pipeline_input"],
        template="Increase the query size to at least 105 characters.\n\nQuery: {pipeline_input}\n",
    )
    augmenter = RunnablePassthrough.assign(
        pipeline_input=DictParser("pipeline_input")
        | augment_prompt
        | llm_client
        | output_parser
    )
    apply_or_augment = RunnableTryFix(
        primary_step=apply_watermark, fix_step=augmenter, log_failures=True
    )
    embed_rag_chain = set_input | apply_or_augment | ragllm
    logging.info("Rizzo2016 chain defined")

    # Evaluation
    logging.info("Starting evaluation")
    eval_fn = partial(
        evaluate,
        metrics=DEFAULT_ALL_METRICS,
        llm=llm_client,
        embeddings=embedding_openai,
    )
    token_rag_results = token_rag_chain.batch(baseline_qas)
    embed_rag_results = embed_rag_chain.batch(baseline_qas)
    token_res = eval_fn(pipeline_results=token_rag_results)
    embed_res = eval_fn(pipeline_results=embed_rag_results)
    res = WLLMKResult(token=token_res, embed=embed_res)
    res.save(f"results/{seed}_wllmk_results.pkl")
    logging.info(f"Pipeline with seed {seed} completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM watermarking pipeline with given seeds."
    )
    parser.add_argument(
        "-s",
        "--seeds",
        metavar="N",
        type=int,
        nargs="+",
        help="an integer for the seed",
        required=True,
    )
    args = parser.parse_args()

    for seed in args.seeds:
        run_pipeline(seed)
