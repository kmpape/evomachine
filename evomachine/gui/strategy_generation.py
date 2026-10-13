"""Background, hardware-free AutoStrat generation using the quick-notebook recipe."""

import asyncio
import multiprocessing as mp
import os
import queue
import time

from autostrat import StrategyPipeline, load_domain_pack
from autostrat.generation import GeneratorConfig, PromptRecipe, StrategyGenerator
from autostrat.verification import SemanticVerifier, SemanticVerifierConfig
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

from evomachine.domain_packs import MICROSCOPY_DOMAIN_PACK_PATH
from evomachine.gui.autostrat_config import (
    ANTHROPIC_PROVIDER,
    OPENAI_COMPATIBLE_PROVIDER,
    ROBIN_OFFICE_2_PRESET,
    AutoStratModelConfig,
)
from evomachine.strategy_generation.preview import GenerationPreview, generate_preview


DEFAULT_GENERATION_TIMEOUT_SECONDS = 300.0
GENERATION_TIMEOUT_ENV = "AUTOSTRAT_GENERATION_TIMEOUT_SECONDS"


def model_config_from_environment() -> AutoStratModelConfig:
    """Retain environment-based configuration for notebooks and non-GUI callers."""
    configured_model_id = os.getenv("AUTOSTRAT_MODEL_ID")
    model_id = (
        ROBIN_OFFICE_2_PRESET.model
        if configured_model_id is None
        else configured_model_id.strip()
    )
    if model_id.startswith("anthropic:"):
        provider = ANTHROPIC_PROVIDER
        model_id = model_id.removeprefix("anthropic:")
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        base_url = os.getenv("ANTHROPIC_BASE_URL")
    else:
        provider = OPENAI_COMPATIBLE_PROVIDER
        model_id = model_id.removeprefix("openai-chat:").removeprefix("openai:")
        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL")
        if configured_model_id is None and base_url is None:
            base_url = ROBIN_OFFICE_2_PRESET.base_url
    return AutoStratModelConfig(
        provider=provider,
        model=model_id,
        api_key=api_key,
        base_url=base_url,
    )


def _create_model(config: AutoStratModelConfig):
    if not config.enabled:
        raise ValueError("Supply an API key before generating an AutoStrat strategy.")
    if config.provider == OPENAI_COMPATIBLE_PROVIDER:
        return OpenAIChatModel(
            config.model,
            provider=OpenAIProvider(
                api_key=config.api_key,
                base_url=config.base_url,
            ),
        )
    return AnthropicModel(
        config.model,
        provider=AnthropicProvider(
            api_key=config.api_key,
            base_url=config.base_url,
        ),
    )


def create_pipeline(model_config: AutoStratModelConfig | None = None):
    config = model_config or model_config_from_environment()
    model_name = config.qualified_model
    model = _create_model(config)
    domain = load_domain_pack(MICROSCOPY_DOMAIN_PACK_PATH)
    generator_config = GeneratorConfig(model=model_name, validation_retries=2)
    verifier_config = SemanticVerifierConfig(model=model_name, output_retries=2)
    return StrategyPipeline(
        domain,
        generator_config=generator_config,
        verifier_config=verifier_config,
        prompt_recipe=PromptRecipe(name="gui", few_shot_count=len(domain.few_shot_examples)),
        semantic_revisions=2,
        generator=StrategyGenerator(generator_config, model=model),
        verifier=SemanticVerifier(verifier_config, model=model),
    )


def generate(
        request: str,
        model_config: AutoStratModelConfig | None = None,
) -> GenerationPreview:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return generate_preview(create_pipeline(model_config), request)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        asyncio.set_event_loop(None)


def _generate_process(
        request: str,
        result_queue,
        model_config: AutoStratModelConfig | None,
) -> None:
    """Generate in a disposable process so cancellation can stop model I/O."""
    try:
        result_queue.put(("preview", generate(request, model_config)))
    except BaseException as error:
        result_queue.put(("error", type(error).__name__, str(error)))


def generation_timeout_seconds() -> float:
    value = os.getenv(GENERATION_TIMEOUT_ENV)
    if value is None:
        return DEFAULT_GENERATION_TIMEOUT_SECONDS
    try:
        timeout = float(value)
    except ValueError as error:
        raise ValueError(f"{GENERATION_TIMEOUT_ENV} must be a positive number.") from error
    if timeout <= 0:
        raise ValueError(f"{GENERATION_TIMEOUT_ENV} must be a positive number.")
    return timeout


def _terminate_process(process) -> None:
    if not process.is_alive():
        return
    process.terminate()
    process.join(timeout=2.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=2.0)


def generate_cancellable(
        request: str,
        cancel_event,
        *,
        model_config: AutoStratModelConfig | None = None,
) -> GenerationPreview:
    """Run generation with cooperative cancellation and a hard process timeout."""
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_generate_process,
        args=(request, result_queue, model_config),
        name="AutoStratGeneration",
        daemon=True,
    )
    started = time.monotonic()
    timeout = generation_timeout_seconds()
    process.start()
    try:
        while process.is_alive():
            if cancel_event.is_set():
                _terminate_process(process)
                return GenerationPreview(
                    elapsed_seconds=time.monotonic() - started,
                    error=RuntimeError("Generation cancelled."),
                )
            if time.monotonic() - started >= timeout:
                _terminate_process(process)
                raise TimeoutError(f"AutoStrat generation exceeded {timeout:g} seconds.")
            process.join(timeout=0.1)
        try:
            message = result_queue.get(timeout=1.0)
        except queue.Empty as error:
            raise RuntimeError(
                f"AutoStrat generation process exited with code {process.exitcode} without a result."
            ) from error
        if message[0] == "preview":
            return message[1]
        raise RuntimeError(f"{message[1]}: {message[2]}")
    finally:
        _terminate_process(process)
        result_queue.close()
        result_queue.join_thread()
