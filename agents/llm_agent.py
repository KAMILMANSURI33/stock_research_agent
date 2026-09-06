from typing import Any, List, Optional, Tuple
from llama_cpp import Llama
from .base_agent import BaseAgent
from config import (
    ARTICLE_CHAR_BUDGET,
    LLAMA_MODEL_PATH,
    MAX_OUTPUT_TOKENS,
    N_CTX,
    PROMPT_SAFETY_MARGIN,
    SUMMARY_PROMPT,
    TEMPERATURE,
)

# Returned instead of a generated summary when no source articles were
# retrieved, so an ungrounded answer is never presented as a real one.
NO_ARTICLES_SUMMARY = (
    "No news articles were retrieved for {symbol}, so no summary was generated."
)


class PromptTooLargeError(RuntimeError):
    """A prompt could not be reduced to fit the model's context window.

    Raised in place of the ValueError llama-cpp throws mid-generation, so the
    caller sees a message naming the budget rather than a library internal.
    """


class LLMAgent(BaseAgent):
    """Agent responsible for LLM-based text processing using Llama."""

    def __init__(self):
        super().__init__("LLM")
        self.model: Optional[Llama] = None

    async def initialize(self) -> None:
        """Initialize the Llama model."""
        self.model = Llama(
            model_path=LLAMA_MODEL_PATH,
            n_ctx=N_CTX,
            n_threads=4
        )

    @staticmethod
    def prompt_token_budget() -> int:
        """Tokens available to the prompt once output and margin are reserved."""
        return N_CTX - MAX_OUTPUT_TOKENS - PROMPT_SAFETY_MARGIN

    def count_tokens(self, text: str) -> int:
        """Token count from the model's own tokenizer, not a heuristic."""
        if not self.model:
            raise RuntimeError("model must be initialized before counting tokens")
        return len(self.model.tokenize(text.encode("utf-8")))

    def build_prompt(
        self, symbol: str, articles: List[str]
    ) -> Tuple[str, List[int], List[str]]:
        """Assemble the largest prompt that fits, and report which articles.

        Each article is first capped at ARTICLE_CHAR_BUDGET. Whole articles are
        then dropped from the end until the tokenized prompt fits the budget, so
        the model never receives an article cut off mid-sentence.

        Returns the indices of the articles actually included, not just a count,
        plus the exact excerpt strings used. Grounding has to be scored against
        the text the model really saw: a figure invented by the model that
        happens to appear in a dropped article would otherwise be counted as
        sourced, and an included-but-truncated article would over-credit the
        tail the model never received.
        """
        budget = self.prompt_token_budget()
        excerpts = [str(a)[:ARTICLE_CHAR_BUDGET] for a in articles]

        for count in range(len(excerpts), 0, -1):
            articles_text = "\n\n".join(
                f"Article {i+1}:\n{a}" for i, a in enumerate(excerpts[:count])
            )
            prompt = SUMMARY_PROMPT.format(symbol=symbol, articles=articles_text)
            if self.count_tokens(prompt) <= budget:
                return prompt, list(range(count)), excerpts[:count]

        # Even a single capped article overflows: refuse rather than let
        # llama-cpp raise ValueError partway through generation.
        smallest = SUMMARY_PROMPT.format(
            symbol=symbol, articles=f"Article 1:\n{excerpts[0]}"
        )
        raise PromptTooLargeError(
            f"Cannot fit even one article within the prompt budget: "
            f"{self.count_tokens(smallest)} tokens needed, {budget} available "
            f"(N_CTX={N_CTX} - MAX_OUTPUT_TOKENS={MAX_OUTPUT_TOKENS} - "
            f"PROMPT_SAFETY_MARGIN={PROMPT_SAFETY_MARGIN}). "
            f"Lower ARTICLE_CHAR_BUDGET (currently {ARTICLE_CHAR_BUDGET})."
        )

    async def process(self, input_data: dict) -> dict:
        """Summarize the supplied articles.

        Returns {"summary", "articles_used", "articles_used_indices",
        "prompt_context"} so the caller can tell exactly which retrieved
        articles reached the model, and with what text.
        """
        symbol = input_data.get('symbol', '')
        articles = input_data.get('articles', [])

        # An empty article set leaves the prompt with no source text, but the
        # model answers it regardless and invents figures to fill the gap.
        # Return early instead, and skip loading the model at all.
        if not articles:
            return {
                "summary": NO_ARTICLES_SUMMARY.format(symbol=symbol),
                "articles_used": 0,
                "articles_used_indices": [],
                "prompt_context": [],
            }

        if not self.model:
            await self.initialize()

        prompt, used_indices, used_excerpts = self.build_prompt(symbol, articles)

        # Generate response
        response = self.model.create_completion(
            prompt,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            stop=["###"]
        )

        return {
            "summary": response['choices'][0]['text'].strip(),
            "articles_used": len(used_indices),
            "articles_used_indices": used_indices,
            # Exact strings that went into the prompt, in prompt order and
            # aligned with articles_used_indices. Carried always; the API layer
            # decides whether to expose it.
            "prompt_context": used_excerpts,
        }

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.model:
            del self.model
            self.model = None
        await super().cleanup()
