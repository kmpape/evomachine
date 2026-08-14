"""Tests for strategy-generation prompt construction."""

import pytest

from evomachine.prompt_constructor import PromptConstructor


def test_construct_prompt_contains_ordered_generation_context() -> None:
    user_request = "Capture an image with a 50 ms exposure on every callback."

    prompt = PromptConstructor(user_request).construct_prompt()

    section_labels = (
        "## DSL GRAMMAR",
        "## DSL SEMANTICS",
        "## EXAMPLES",
        "## USER REQUEST",
        "## RESPONSE REQUIREMENTS",
    )
    positions = [prompt.index(label) for label in section_labels]
    assert positions == sorted(positions)
    assert user_request in prompt
    assert 'FOV_TARGET: "first_fov" | "next_fov"' in prompt
    assert "Return only the complete DSL strategy as plain text." in prompt
    assert "request is nonsensical or cannot be represented" in prompt
    assert "include only the initialise, callback, and finalise section headings" in prompt


def test_construct_prompt_is_idempotent() -> None:
    constructor = PromptConstructor("Take a 100 ms image.")

    first_prompt = constructor.construct_prompt()
    second_prompt = constructor.construct_prompt()

    assert second_prompt == first_prompt
    assert second_prompt.count("## DSL GRAMMAR") == 1


def test_constructor_accepts_custom_prompt_components() -> None:
    prompt = PromptConstructor(
        "Generate a strategy.",
        grammar_text="start: \"custom\"",
        semantic_guidance_text="Custom semantics.",
        few_shot_examples_text="Custom examples.",
    ).construct_prompt()

    assert 'start: "custom"' in prompt
    assert "Custom semantics." in prompt
    assert "Custom examples." in prompt


def test_constructor_rejects_non_string_user_input() -> None:
    with pytest.raises(TypeError, match="nat_lang_input must be str"):
        PromptConstructor(123)  # type: ignore[arg-type]
