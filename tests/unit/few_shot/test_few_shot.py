"""
Unit tests for few-shot example support (issue #32).
"""

import pytest
import tiktoken

from delm import DELMConfig, Schema
from delm.models import ExtractionVariable
from delm.utils.few_shot import (
    FewShotExampleSelector,
    validate_few_shot_params,
)


@pytest.fixture
def examples():
    return [
        {
            "text": "Goldman Sachs raised its oil price target.",
            "output": {"company": "Goldman Sachs"},
        },
        {
            "text": "Barclays sees weakness in gas markets.",
            "output": {"company": "Barclays"},
        },
        {
            "text": "Deutsche Bank upgraded steel producers.",
            "output": {"company": "Deutsche Bank"},
        },
        {
            "text": "JP Morgan cut its copper forecast.",
            "output": {"company": "JP Morgan"},
        },
    ]


@pytest.fixture
def simple_schema():
    return Schema.simple(
        variables_list=[
            ExtractionVariable(
                name="company",
                description="Company name",
                data_type="string",
            )
        ]
    )


class TestFewShotExampleSelector:
    def test_selects_first_n_when_not_random(self, examples):
        selector = FewShotExampleSelector(examples, num_examples=2)
        selected = selector.select_examples()
        assert selected == examples[:2]

    def test_num_examples_capped_at_pool_size(self, examples):
        selector = FewShotExampleSelector(examples, num_examples=10)
        assert len(selector.select_examples()) == len(examples)

    def test_random_sampling_is_reproducible(self, examples):
        selector_a = FewShotExampleSelector(
            examples, num_examples=2, random_sample=True, seed=42
        )
        selector_b = FewShotExampleSelector(
            examples, num_examples=2, random_sample=True, seed=42
        )
        draws_a = [
            tuple(e["text"] for e in selector_a.select_examples()) for _ in range(5)
        ]
        draws_b = [
            tuple(e["text"] for e in selector_b.select_examples()) for _ in range(5)
        ]
        assert draws_a == draws_b

    def test_random_sampling_draws_from_pool(self, examples):
        selector = FewShotExampleSelector(examples, num_examples=2, random_sample=True)
        for _ in range(10):
            selected = selector.select_examples()
            assert len(selected) == 2
            for example in selected:
                assert example in examples

    def test_truncation_limits_example_tokens(self, examples):
        long_example = [{"text": "word " * 500, "output": {"company": "X"}}]
        selector = FewShotExampleSelector(
            long_example, num_examples=1, truncate_length=10
        )
        block = selector.build_examples_block()
        text_line = [l for l in block.splitlines() if l.startswith("Text: ")][0]
        tokenizer = tiktoken.get_encoding("cl100k_base")
        assert len(tokenizer.encode(text_line.removeprefix("Text: "))) <= 10

    def test_no_truncation_by_default(self, examples):
        selector = FewShotExampleSelector(examples, num_examples=1)
        block = selector.build_examples_block()
        assert examples[0]["text"] in block

    def test_examples_block_contains_outputs_as_json(self, examples):
        selector = FewShotExampleSelector(examples, num_examples=2)
        block = selector.build_examples_block()
        assert '"company": "Goldman Sachs"' in block
        assert "Example 1:" in block
        assert "Example 2:" in block

    def test_prepend_adds_examples_block_once(self, examples):
        selector = FewShotExampleSelector(examples, num_examples=2)
        prompt = "Extract - company from:\nchunk text"
        result = selector.prepend_to_prompt(prompt)
        assert result.startswith("Examples:")
        assert result.endswith(prompt)
        assert result.count("Examples:") == 1
        assert '"company": "Goldman Sachs"' in result

    def test_from_optional_returns_none_without_examples(self):
        selector = FewShotExampleSelector.from_optional(
            examples=None,
            num_examples=3,
            truncate_length=None,
            random_sample=False,
        )
        assert selector is None


class TestFewShotWithMultipleSchema:
    def test_examples_block_appears_once_for_multiple_schema(self, examples):
        """MultipleSchema repeats the template per sub-schema; the examples
        block must still appear exactly once in the final prompt."""
        multiple_schema = Schema.multiple(
            companies=Schema.simple(
                variables_list=[
                    ExtractionVariable(
                        name="company",
                        description="Company name",
                        data_type="string",
                    )
                ]
            ),
            prices=Schema.simple(
                variables_list=[
                    ExtractionVariable(
                        name="price",
                        description="Price value",
                        data_type="number",
                    )
                ]
            ),
        )
        selector = FewShotExampleSelector(examples, num_examples=2)
        prompt = multiple_schema.schema.create_prompt(
            "chunk text",
            "Extract the following information from the text:\n\n{variables}\n\nText to analyze:\n{text}",
        )
        prompt = selector.prepend_to_prompt(prompt)
        assert prompt.count("Examples:") == 1
        assert prompt.count("## COMPANIES") == 1
        assert prompt.count("## PRICES") == 1


class TestFewShotValidation:
    def test_rejects_empty_examples(self):
        with pytest.raises(ValueError, match="non-empty list"):
            validate_few_shot_params([], 1, None)

    def test_rejects_example_missing_output(self):
        with pytest.raises(ValueError, match="missing keys"):
            validate_few_shot_params([{"text": "abc"}], 1, None)

    def test_rejects_example_missing_text(self):
        with pytest.raises(ValueError, match="missing keys"):
            validate_few_shot_params([{"output": {}}], 1, None)

    def test_rejects_non_dict_example(self):
        with pytest.raises(ValueError, match="must be a dict"):
            validate_few_shot_params(["not a dict"], 1, None)

    def test_rejects_non_positive_num_examples(self):
        examples = [{"text": "a", "output": {}}]
        with pytest.raises(ValueError, match="positive integer"):
            validate_few_shot_params(examples, 0, None)

    def test_rejects_non_positive_truncate_length(self):
        examples = [{"text": "a", "output": {}}]
        with pytest.raises(ValueError, match="positive integer or None"):
            validate_few_shot_params(examples, 1, -5)


class TestFewShotConfig:
    def test_default_config_has_no_few_shot(self, simple_schema):
        config = DELMConfig(schema=simple_schema)
        assert config.llm_extraction_cfg.few_shot_examples is None

    def test_config_validation_accepts_valid_few_shot(self, simple_schema, examples):
        config = DELMConfig(
            schema=simple_schema,
            few_shot_examples=examples,
            few_shot_num_examples=2,
            few_shot_truncate_length=100,
            few_shot_random_sample=True,
        )
        config.validate()

    def test_config_validation_rejects_bad_examples(self, simple_schema):
        config = DELMConfig(
            schema=simple_schema,
            few_shot_examples=[{"text": "missing output"}],
        )
        with pytest.raises(ValueError, match="missing keys"):
            config.validate()

    def test_config_serialization_round_trip(self, simple_schema, examples):
        config = DELMConfig(
            schema=simple_schema,
            few_shot_examples=examples,
            few_shot_num_examples=2,
            few_shot_truncate_length=64,
            few_shot_random_sample=True,
        )
        restored = DELMConfig.from_dict(config.to_dict())
        llm_cfg = restored.llm_extraction_cfg
        assert llm_cfg.few_shot_examples == examples
        assert llm_cfg.few_shot_num_examples == 2
        assert llm_cfg.few_shot_truncate_length == 64
        assert llm_cfg.few_shot_random_sample is True
