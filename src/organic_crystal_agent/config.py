import os

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Runtime configuration for the organic crystal agent."""

    objective: str = Field(
        default="Propose the single most informative next organic crystal experiment."
    )
    max_candidates: int = Field(default=3, ge=1, le=20)
    default_batch_size: int = Field(default=1, ge=1, le=8)
    enable_verification: bool = True
    research_backend: str = "auto"
    # NOTE: Defaults must be valid OpenAI model ids; otherwise molecule proposal silently stalls.
    # Keep these cheap by default; override via env vars when needed.
    openai_model: str = "gpt-5.5-mini"
    openai_molecule_model: str = "gpt-5.5"
    openai_main_planner_model: str = "gpt-5.5-mini"
    openai_literature_model: str = "gpt-5.5"
    openai_research_model: str = "gpt-5.5-mini"


def resolve_openai_model_env(
    specific_env: str,
    *,
    specific_default: str,
    general_env: str = "ORGANIC_CRYSTAL_AGENT_OPENAI_MODEL",
    general_default: str = "gpt-5.5-mini",
) -> str:
    """Resolve model overrides with cheap general defaults and role-specific opt-ins."""

    model = os.getenv(
        specific_env,
        os.getenv(general_env, specific_default or general_default),
    )

    # Guardrail: some deployments set literal placeholder model ids such as
    # "gpt-5.x". Real model ids like "gpt-5.4" should pass through.
    if model and model.strip().lower() in {"gpt-5", "gpt-5.x", "gpt-5-mini", "gpt-5-nano"}:
        return "gpt-5.5-mini"

    return model
