"""Background, hardware-free AutoStrat generation using the quick-notebook recipe."""

import asyncio
import multiprocessing as mp
import os
import queue
import time

from autostrat import StrategyPipeline, load_domain_pack
from autostrat.generation import GeneratorConfig, PromptRecipe
from autostrat.verification import SemanticVerifierConfig

from evomachine.domain_packs import MICROSCOPY_DOMAIN_PACK_PATH
from evomachine.strategy_generation.preview import GenerationPreview, generate_preview


DEFAULT_GENERATION_TIMEOUT_SECONDS = 300.0
GENERATION_TIMEOUT_ENV = "AUTOSTRAT_GENERATION_TIMEOUT_SECONDS"


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


def _generate_process(request: str, result_queue) -> None:
    """Generate in a disposable process so cancellation can stop model I/O."""
    try:
        result_queue.put(("preview", generate(request)))
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


def generate_cancellable(request: str, cancel_event) -> GenerationPreview:
    """Run generation with cooperative cancellation and a hard process timeout."""
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_generate_process,
        args=(request, result_queue),
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
