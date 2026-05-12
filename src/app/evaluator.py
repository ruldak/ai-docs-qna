import json
from typing import List
from llama_index.core.llms import ChatMessage

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


class LlamaIndexEvaluator:
    def __init__(self, llm):
        self.llm = llm

    def _call_judge(self, prompt: str) -> dict:
        messages = [
            ChatMessage(
                role="system",
                content="Kamu adalah AI evaluator. Selalu output JSON valid."
            ),
            ChatMessage(role="user", content=prompt)
        ]

        response = self.llm.chat(messages)
        content = response.message.content.strip()

        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()
        return json.loads(content)

    def evaluate_faithfulness(self, question: str, answer: str, contexts: List[str]) -> dict:
        contexts_text = "\n\n".join(
            [f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)]
        )

        prompt = FAITHFULNESS_PROMPT.format(
            contexts=contexts_text,
            question=question,
            answer=answer
        )

        return self._call_judge(prompt)

    def evaluate_relevancy(self, question: str, answer: str) -> dict:
        prompt = RELEVANCY_PROMPT.format(
            question=question,
            answer=answer
        )

        return self._call_judge(prompt)

    def evaluate(self, question: str, answer: str, contexts: List[str]) -> dict:
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