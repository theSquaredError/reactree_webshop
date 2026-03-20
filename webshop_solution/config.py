from dataclasses import dataclass, field


@dataclass
class LLMAgentConfig:
	# model_name: str = "my-llama3-gguf"
	model_name:str = "/Users/vikas/Downloads/llama3-local/model.gguf"
	backend: str = "guidance_llamacpp"  # ollama | openai
	temperature: float = 0.4
	max_tokens: int = 256
	ollama_url: str = "http://localhost:11434/api/generate"
	openai_api_key: str | None = None,
	n_ctx = 8192


@dataclass
class PlannerConfig:
	max_steps: int = 20
	max_decisions: int = 40
	max_depth: int = 100
	num_products: int = 100
	random_seed: int = 0
	collect_dir: str = "webshop_trajectories"


@dataclass
class WebShopReAcTreeConfig:
	llm_agent: LLMAgentConfig = field(default_factory=LLMAgentConfig)
	planner: PlannerConfig = field(default_factory=PlannerConfig)


def default_config() -> WebShopReAcTreeConfig:
	return WebShopReAcTreeConfig()

