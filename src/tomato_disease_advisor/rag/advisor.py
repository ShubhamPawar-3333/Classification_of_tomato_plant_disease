"""
Treatment Advisor (LLM-powered)

Uses retrieved knowledge chunks + disease classification results to
generate a natural language treatment advisory via Groq LLM.

Combines:
  - Disease information (from knowledge/diseases/)
  - Treatment options (from knowledge/treatments/)
  - Classification confidence and severity level
into a structured, actionable advisory for the farmer.
"""
import os
from typing import Dict, List, Optional

from tomato_disease_advisor.entity import RAGConfig


# System prompt template for the LLM
SYSTEM_PROMPT = """You are an expert agricultural advisor specializing in tomato diseases.
Based on the disease diagnosis and relevant knowledge provided, give a clear,
actionable treatment advisory for the farmer.

Guidelines:
- Be specific with product names and dosages
- Include both chemical AND organic/biological options
- Mention urgency level (immediate action vs. routine)
- Add preventive measures for future crops
- Keep language simple and practical
- If confidence is low, mention that a field visit by an expert is recommended
- Structure your response with clear sections"""

# User prompt template
USER_PROMPT_TEMPLATE = """## Disease Diagnosis
- **Disease:** {disease_name}
- **Confidence:** {confidence:.1%}
- **Severity:** {severity_level} ({affected_area:.1f}% affected area)

## Relevant Disease Information
{disease_context}

## Available Treatment Options
{treatment_context}

---

Based on the above diagnosis and knowledge, provide a treatment advisory.
Include: immediate actions, recommended sprays with dosages, organic alternatives,
and preventive measures. Consider the severity level when recommending urgency."""


class TreatmentAdvisor:
    """
    Generates LLM-powered treatment advisories using RAG.

    Flow: Classification result + retrieved knowledge → LLM → advisory text
    """

    def __init__(self, config: RAGConfig):
        """
        Initialize TreatmentAdvisor.

        Args:
            config: RAGConfig with LLM provider, model, and generation settings
        """
        self.config = config
        self._client = None

    def _get_client(self):
        """Lazy-initialize the LLM client (Groq)."""
        if self._client is not None:
            return self._client

        if self.config.llm_provider.lower() == "groq":
            try:
                from groq import Groq
            except ImportError:
                raise ImportError("groq package required: pip install groq")

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError(
                    "GROQ_API_KEY environment variable is not set. "
                    "Get a free key at https://console.groq.com/"
                )

            self._client = Groq(api_key=api_key)
            print(f"[Advisor] Groq client initialized "
                  f"(model: {self.config.llm_model})")
        else:
            raise ValueError(
                f"Unsupported LLM provider: {self.config.llm_provider}. "
                f"Currently supported: groq"
            )

        return self._client

    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved knowledge chunks into context string."""
        if not chunks:
            return "No specific information available."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source", "unknown")
            score = chunk.get("score", 0.0)
            content = chunk.get("content", "")
            context_parts.append(
                f"### Source {i}: {source} (relevance: {score:.2f})\n{content}"
            )

        return "\n\n".join(context_parts)

    def generate_advisory(
        self,
        disease_name: str,
        confidence: float,
        severity_level: str,
        affected_area: float,
        disease_chunks: List[Dict],
        treatment_chunks: List[Dict],
    ) -> Dict:
        """
        Generate a treatment advisory using the LLM.

        Args:
            disease_name: Predicted disease class name
            confidence: Model confidence (0-1)
            severity_level: Severity level string
            affected_area: Affected area percentage
            disease_chunks: Retrieved disease info chunks
            treatment_chunks: Retrieved treatment info chunks

        Returns:
            dict with: advisory (str), model_used (str), tokens_used (int)
        """
        client = self._get_client()

        # Build context from retrieved chunks
        disease_context = self._format_context(disease_chunks)
        treatment_context = self._format_context(treatment_chunks)

        # Format user prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            disease_name=disease_name,
            confidence=confidence,
            severity_level=severity_level,
            affected_area=affected_area,
            disease_context=disease_context,
            treatment_context=treatment_context,
        )

        # Call LLM
        try:
            response = client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            advisory_text = response.choices[0].message.content
            tokens_used = getattr(response.usage, "total_tokens", 0)

            print(f"[Advisor] Advisory generated "
                  f"({tokens_used} tokens used)")

            return {
                "advisory": advisory_text,
                "model_used": self.config.llm_model,
                "tokens_used": tokens_used,
                "disease_name": disease_name,
            }

        except Exception as e:
            print(f"[Advisor] LLM error: {e}")
            # Return a fallback advisory from the retrieved knowledge
            return self._fallback_advisory(
                disease_name, severity_level, treatment_chunks
            )

    def _fallback_advisory(
        self,
        disease_name: str,
        severity_level: str,
        treatment_chunks: List[Dict],
    ) -> Dict:
        """
        Generate a simple fallback advisory when LLM is unavailable.

        Uses raw treatment knowledge without LLM processing.
        """
        if treatment_chunks:
            raw_treatment = treatment_chunks[0].get("content", "")
            advisory = (
                f"## Treatment Advisory for {disease_name}\n\n"
                f"**Severity:** {severity_level}\n\n"
                f"**Note:** AI advisory is currently unavailable. "
                f"Below is the relevant treatment information from our "
                f"knowledge base:\n\n{raw_treatment}"
            )
        else:
            advisory = (
                f"## Treatment Advisory for {disease_name}\n\n"
                f"**Severity:** {severity_level}\n\n"
                f"No specific treatment information available. "
                f"Please consult a local agricultural extension officer "
                f"for treatment recommendations."
            )

        return {
            "advisory": advisory,
            "model_used": "fallback",
            "tokens_used": 0,
            "disease_name": disease_name,
        }
