"""
RAG helper for the Module 1 demo: the same RAGBase class as the course,
adapted to PubMed abstracts and to the Anthropic SDK.

    search  -> build_context -> build_prompt -> llm
    |________________________ rag ________________________|
"""

import re
from dataclasses import dataclass

import anthropic

MODEL = "claude-haiku-5-5"

# Claude Haiku 5.5 list price, USD per 1M tokens (prompts up to 100K tokens)
PRICE_INPUT = 0.10
PRICE_OUTPUT = 0.50

INSTRUCTIONS = """
You are a research assistant helping with a literature review on
AI for early lung cancer detection from CT.

Answer the question using only the abstracts provided in the context.
After each claim, cite the supporting abstract(s) by PMID in square
brackets, like [PMID 12345678]. Prefer naming the first author and year
when describing who did what.

If the context does not contain the information needed, say
"Not covered in the retrieved abstracts." Do not use outside knowledge.
""".strip()

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


@dataclass
class RAGResult:
    answer: str
    input_tokens: int
    output_tokens: int
    search_results: list

    @property
    def cost_usd(self):
        return (self.input_tokens * PRICE_INPUT + self.output_tokens * PRICE_OUTPUT) / 1_000_000


def cited_pmids(text):
    """PMIDs the model cited, e.g. '[PMID 12345678]' or '[PMID 123, 456]'."""
    found = []
    for group in re.findall(r"\[PMID[^\]]*\]", text):
        found.extend(re.findall(r"\d{6,9}", group))
    return sorted(set(found))


class RAGBase:
    def __init__(
        self,
        index,
        llm_client=None,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model=MODEL,
        year_from=None,
        year_to=None,
    ):
        self.index = index
        self.llm_client = llm_client or anthropic.Anthropic()
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model
        self.year_from = year_from
        self.year_to = year_to

    def search(self, query, num_results=5):
        boost_dict = {"title": 2.0, "abstract": 1.0}

        filter_dict = {}
        year_range = []
        if self.year_from is not None:
            year_range.append((">=", self.year_from))
        if self.year_to is not None:
            year_range.append(("<=", self.year_to))
        if year_range:
            filter_dict["year"] = year_range

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
        )

    def build_context(self, search_results):
        blocks = []
        for doc in search_results:
            blocks.append(
                f"PMID {doc['pmid']} | {doc['first_author']} et al., {doc['year']} | {doc['journal']}\n"
                f"TITLE: {doc['title']}\n"
                f"ABSTRACT: {doc['abstract']}"
            )
        return "\n\n".join(blocks)

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt):
        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=self.instructions,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "low"},  # short factual answers; keeps thinking cheap
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return text, response.usage

    def rag(self, query, num_results=5):
        search_results = self.search(query, num_results=num_results)
        prompt = self.build_prompt(query, search_results)
        answer, usage = self.llm(prompt)
        return RAGResult(
            answer=answer,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            search_results=search_results,
        )
