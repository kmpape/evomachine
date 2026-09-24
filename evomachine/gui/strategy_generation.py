"""Background, hardware-free AutoStrat generation using the quick-notebook recipe."""

import asyncio
import os

from autostrat import StrategyPipeline, load_domain_pack
from autostrat.generation import GeneratorConfig, PromptRecipe
from autostrat.verification import SemanticVerifierConfig

from evomachine.domain_packs import MICROSCOPY_DOMAIN_PACK_PATH
from evomachine.strategy_generation.preview import GenerationPreview, generate_preview


def create_pipeline():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in the GUI backend environment before generating.")
    os.environ.setdefault("OPENAI_BASE_URL", "https://robin-office-2.tail32bb7.ts.net/v1")
    model_id = os.getenv("AUTOSTRAT_MODEL_ID", "qwen3.6")
    model = model_id if ":" in model_id else f"openai-chat:{model_id}"
    domain = load_domain_pack(MICROSCOPY_DOMAIN_PACK_PATH)
    return StrategyPipeline(
        domain,
        generator_config=GeneratorConfig(model=model, validation_retries=2),
        verifier_config=SemanticVerifierConfig(model=model, output_retries=2),
        prompt_recipe=PromptRecipe(name="gui", few_shot_count=len(domain.few_shot_examples)),
        semantic_revisions=2,
    )


def generate(request: str) -> GenerationPreview:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return generate_preview(create_pipeline(), request)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        asyncio.set_event_loop(None)
