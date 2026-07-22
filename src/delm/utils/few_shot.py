"""Few-shot example selection and prompt injection for DELM.

Renders hand-labeled examples into the extraction prompt. Supports limiting
the number of examples, truncating example text to a token budget, and
random sampling from the provided ground-truth pool.
"""

import json
import logging
import random
from typing import Any, Dict, List, Optional

import tiktoken

from delm.constants import SYSTEM_RANDOM_SEED

log = logging.getLogger(__name__)

FEW_SHOT_PLACEHOLDER = "{examples}"


class FewShotExampleSelector:
    """Select and render few-shot examples for extraction prompts.

    Each example is a mapping with a ``"text"`` key (source text) and an
    ``"output"`` key (expected extraction result as a dict or JSON string).
    """

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        num_examples: int,
        truncate_length: Optional[int] = None,
        random_sample: bool = False,
        seed: int = SYSTEM_RANDOM_SEED,
    ) -> None:
        """Initialize the selector.

        Args:
            examples: Pool of ground-truth examples, each with ``"text"`` and
                ``"output"`` keys.
            num_examples: Number of examples to include per prompt. Capped at
                the pool size.
            truncate_length: Maximum token length for each example's text.
                ``None`` disables truncation.
            random_sample: Whether to randomly sample examples from the pool
                for each prompt. When False, the first ``num_examples``
                examples are always used.
            seed: Seed for the sampling RNG (reproducible across runs).
        """
        validate_few_shot_params(examples, num_examples, truncate_length)

        self.examples = examples
        self.num_examples = min(num_examples, len(examples))
        if self.num_examples < num_examples:
            log.warning(
                "Requested %d few-shot examples but only %d available; using %d",
                num_examples,
                len(examples),
                self.num_examples,
            )
        self.truncate_length = truncate_length
        self.random_sample = random_sample
        self._rng = random.Random(seed)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    @classmethod
    def from_optional(
        cls,
        examples: Optional[List[Dict[str, Any]]],
        num_examples: int,
        truncate_length: Optional[int],
        random_sample: bool,
    ) -> Optional["FewShotExampleSelector"]:
        """Build a selector, or return None when no examples are configured."""
        if examples is None:
            return None
        return cls(
            examples=examples,
            num_examples=num_examples,
            truncate_length=truncate_length,
            random_sample=random_sample,
        )

    def select_examples(self) -> List[Dict[str, Any]]:
        """Return the examples to include in the next prompt."""
        if self.random_sample:
            return self._rng.sample(self.examples, self.num_examples)
        return self.examples[: self.num_examples]

    def build_examples_block(self) -> str:
        """Render the selected examples as a text block for the prompt."""
        lines: List[str] = ["Examples:"]
        for i, example in enumerate(self.select_examples(), start=1):
            text = self._truncate(str(example["text"]))
            output = example["output"]
            output_json = output if isinstance(output, str) else json.dumps(output)
            lines.append(f"\nExample {i}:")
            lines.append(f"Text: {text}")
            lines.append(f"Output: {output_json}")
        return "\n".join(lines)

    def inject_into_template(self, prompt_template: str) -> str:
        """Insert the rendered examples block into a prompt template.

        If the template contains the ``{examples}`` placeholder, it is
        replaced in place; otherwise the block is prepended to the template.
        Braces inside the examples (e.g. JSON outputs) are escaped so the
        template can still be passed through ``str.format``.

        Args:
            prompt_template: Extraction prompt template with ``{variables}``
                and ``{text}`` placeholders.

        Returns:
            The template with the examples block included.
        """
        examples_block = (
            self.build_examples_block().replace("{", "{{").replace("}", "}}")
        )
        if FEW_SHOT_PLACEHOLDER in prompt_template:
            return prompt_template.replace(FEW_SHOT_PLACEHOLDER, examples_block)
        return f"{examples_block}\n\n{prompt_template}"

    def _truncate(self, text: str) -> str:
        if self.truncate_length is None:
            return text
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= self.truncate_length:
            return text
        return self._tokenizer.decode(tokens[: self.truncate_length])


def validate_few_shot_params(
    examples: List[Dict[str, Any]],
    num_examples: int,
    truncate_length: Optional[int],
) -> None:
    """Validate few-shot configuration values.

    Args:
        examples: Pool of ground-truth examples.
        num_examples: Number of examples to include per prompt.
        truncate_length: Maximum token length per example text or None.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if not isinstance(examples, list) or not examples:
        raise ValueError(
            f"few_shot_examples must be a non-empty list of dicts. "
            f"few_shot_examples: {examples}"
        )
    for i, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(
                f"few_shot_examples[{i}] must be a dict with 'text' and 'output' "
                f"keys, got {type(example).__name__}"
            )
        missing = {"text", "output"} - set(example)
        if missing:
            raise ValueError(
                f"few_shot_examples[{i}] is missing keys: {sorted(missing)}. "
                f"Each example needs 'text' and 'output'."
            )
    if not isinstance(num_examples, int) or num_examples <= 0:
        raise ValueError(
            f"few_shot_num_examples must be a positive integer. "
            f"few_shot_num_examples: {num_examples}"
        )
    if truncate_length is not None and (
        not isinstance(truncate_length, int) or truncate_length <= 0
    ):
        raise ValueError(
            f"few_shot_truncate_length must be a positive integer or None. "
            f"few_shot_truncate_length: {truncate_length}"
        )
