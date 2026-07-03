import json
import re
from typing import List, Optional
from llama_index.core.llms import ChatMessage
from src.constants import settings
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# PROMPTS
# ============================================================================

FAITHFULNESS_PROMPT = """Kamu adalah evaluator kualitas AI yang ketat dan objektif. 
Tugasmu menilai sejauh mana jawaban berikut didukung oleh konteks yang diberikan.
Jawaban TIDAK BOLEH mengandung informasi yang tidak ada di konteks (hallucination).

KONTEKS:
{contexts}

PERTANYAAN USER: {question}
JAWABAN AI: {answer}

Beri nilai FAITHFULNESS antara 0.0 (tidak ada dukungan sama sekali, banyak hallucination) 
sampai 1.0 (100% didukung konteks, zero hallucination). 
Berikan juga penjelasan singkat dalam Bahasa Indonesia.

Output HARUS dalam format JSON valid:
{{
  "score": 0.95,
  "reasoning": "Jawaban didukung oleh konteks poin 1 dan 3, tidak ada informasi yang dibuat-buat."
}}
"""

RELEVANCY_PROMPT = """Kamu adalah evaluator kualitas AI yang ketat dan objektif.
Tugasmu menilai sejauh mana jawaban relevan dengan pertanyaan yang diajukan.
Jawaban yang baik langsung menjawab pertanyaan tanpa informasi yang tidak diminta.

PERTANYAAN USER: {question}
JAWABAN AI: {answer}

Beri nilai RELEVANCY antara 0.0 (sama sekali tidak relevan) sampai 1.0 
(sangat relevan, tepat, dan komprehensif). 
Berikan juga penjelasan singkat dalam Bahasa Indonesia.

Output HARUS dalam format JSON valid:
{{
  "score": 0.88,
  "reasoning": "Jawaban langsung menjawab pertanyaan tanpa informasi yang tidak diminta."
}}
"""


class PromptInjectionError(Exception):
    """Raised when potential prompt injection is detected."""
    pass


def sanitize_for_prompt(text: str, max_length: int = 10000) -> str:
    """
    Sanitize text for LLM prompts.
    - Escape curly braces to prevent format string issues
    - Limit length
    - Remove null bytes
    """
    if not text:
        return ""

    text = text.replace("\x00", "")

    text = text.replace("{", "{{").replace("}", "}}")

    if len(text) > max_length:
        text = text[:max_length] + "\n...[truncated]"

    return text


def validate_json_output(content: str) -> dict:
    """
    Validate and parse JSON output from the LLM.
    Handle common formatting issues.
    """
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.error(f"Failed to parse JSON: {content[:200]}...")
        raise ValueError(f"Invalid JSON response: {e}")


class LlamaIndexEvaluator:
    """Evaluator using LlamaIndex LLM to assess RAG quality."""

    def __init__(self, llm):
        self.llm = llm

    def _call_judge(self, prompt: str) -> dict:
        """
        Call the judge LLM with proper error handling.

        Args:
            prompt: Sanitized prompt

        Returns:
            Parsed JSON response
        """
        messages = [
            ChatMessage(
                role="system",
                content="Kamu adalah AI evaluator. Selalu output JSON valid."
            ),
            ChatMessage(role="user", content=prompt)
        ]

        try:
            response = self.llm.chat(messages)
            content = response.message.content.strip()
            return validate_json_output(content)
        except Exception as e:
            logger.error(f"Judge LLM call failed: {e}")
            raise

    def evaluate_faithfulness(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str]
    ) -> dict:
        """
        Evaluate the faithfulness of the answer against the context.

        Args:
            question: User question
            answer: AI answer
            contexts: List of retrieved contexts

        Returns:
            Dict with score and reasoning
        """
        safe_question = sanitize_for_prompt(question)
        safe_answer = sanitize_for_prompt(answer)

        contexts_text = "\n\n".join(
            [f"[{i+1}] {sanitize_for_prompt(ctx)}" for i, ctx in enumerate(contexts)]
        )

        prompt = FAITHFULNESS_PROMPT.format(
            contexts=contexts_text,
            question=safe_question,
            answer=safe_answer
        )

        return self._call_judge(prompt)

    def evaluate_relevancy(self, question: str, answer: str) -> dict:
        """
        Evaluate the relevance of the answer to the question.

        Args:
            question: User question
            answer: AI answer

        Returns:
            Dict with score and reasoning
        """
        safe_question = sanitize_for_prompt(question)
        safe_answer = sanitize_for_prompt(answer)

        prompt = RELEVANCY_PROMPT.format(
            question=safe_question,
            answer=safe_answer
        )

        return self._call_judge(prompt)

    def evaluate(
        self, 
        question: str, 
        answer: str, 
        contexts: List[str]
    ) -> dict:
        """
        Complete evaluation: faithfulness + relevancy.

        Args:
            question: User question
            answer: AI answer
            contexts: List of retrieved contexts

        Returns:
            Dict with all scores and reasoning
        """
        faithfulness = self.evaluate_faithfulness(question, answer, contexts)
        relevancy = self.evaluate_relevancy(question, answer)

        return {
            "faithfulness_score": faithfulness.get("score"),
            "faithfulness_reasoning": faithfulness.get("reasoning"),
            "answer_relevancy_score": relevancy.get("score"),
            "answer_relevancy_reasoning": relevancy.get("reasoning"),
            "raw_faithfulness": faithfulness,
            "raw_relevancy": relevancy,
        }