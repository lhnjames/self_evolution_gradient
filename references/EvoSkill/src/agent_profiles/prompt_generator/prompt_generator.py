from claude_agent_sdk import ClaudeAgentOptions
from src.schemas import PromptGeneratorResponse
from src.agent_profiles.prompt_generator.prompt import PROMPT_GENERATOR_SYSTEM_PROMPT
from src.agent_profiles.skill_generator import get_project_root


PROMPT_GENERATOR_TOOLS = ["Read", "Bash", "Glob", "Grep", "WebFetch", "WebSearch", "TodoWrite", "BashOutput"]


prompt_generator_system_prompt = {
    "type": "preset",
    "preset": "claude_code",
    "append": PROMPT_GENERATOR_SYSTEM_PROMPT.strip()
}

prompt_generator_output_format = {
    "type": "json_schema",
    "schema": PromptGeneratorResponse.model_json_schema()
}

prompt_generator_options = ClaudeAgentOptions(
    output_format=prompt_generator_output_format,
    system_prompt=prompt_generator_system_prompt,
    allowed_tools=PROMPT_GENERATOR_TOOLS,
    cwd=get_project_root(),
)
