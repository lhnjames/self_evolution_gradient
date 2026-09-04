from claude_agent_sdk import ClaudeAgentOptions
from src.schemas import PromptProposerResponse
from src.agent_profiles.prompt_proposer.prompt import PROMPT_PROPOSER_SYSTEM_PROMPT
from src.agent_profiles.skill_generator import get_project_root


PROMPT_PROPOSER_TOOLS = [
    "Read",
    "Bash",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "BashOutput",
]


prompt_proposer_system_prompt = {
    "type": "preset",
    "preset": "claude_code",
    "append": PROMPT_PROPOSER_SYSTEM_PROMPT.strip(),
}

prompt_proposer_output_format = {
    "type": "json_schema",
    "schema": PromptProposerResponse.model_json_schema(),
}

prompt_proposer_options = ClaudeAgentOptions(
    output_format=prompt_proposer_output_format,
    system_prompt=prompt_proposer_system_prompt,
    allowed_tools=PROMPT_PROPOSER_TOOLS,
    cwd=get_project_root(),
)
