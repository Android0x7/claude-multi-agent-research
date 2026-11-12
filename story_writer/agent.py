"""Entry point for research agent using AgentDefinition for subagents."""

import asyncio
import os
from pathlib import Path
from ddgs import DDGS
from dotenv import load_dotenv
from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeSDKClient, ClaudeAgentOptions, AgentDefinition, HookMatcher

from story_writer.utils.subagent_tracker import SubagentTracker
from story_writer.utils.transcript import setup_session, TranscriptWriter
from story_writer.utils.message_handler import process_assistant_message

#from duckduckgo_search import DDGS
import httpx
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud")
OLLAMA_MODEL_LOCAL = os.environ.get("OLLAMA_MODEL_LOCAL", "qwen3:0.6b")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama.com")

# Paths to prompt files
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(filename: str) -> str:
    """Load a prompt from the prompts directory."""
    prompt_path = PROMPTS_DIR / filename
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

   # 2. Define the web search tool
@tool(
    "WebSearchCustom", 
    "Searches the web for current information, news, and facts.", 
    {"query": str}
)
async def WebSearchCustom(args: dict) -> dict:
    print(" Searching the web ...")
    query = args.get("query", "")
    if not query:
        return {"content": [{"type": "text", "text": "No search query provided."}]}
        
    try:
        results = []
        # DDGS handles rate limits, user-agents, and structure updates natively
        with DDGS() as ddgs:
            # Fetch the top 4 text-based search results
            response = ddgs.text(query, max_results=4)
            
            for i, res in enumerate(response):
                title = res.get("title", "No Title")
                snippet = res.get("body", "No Description")
                link = res.get("href", "")
                
                results.append(f"Result {i+1}: {title}\nURL: {link}\nSnippet: {snippet}")
                
        search_data = "\n\n".join(results) if results else "No clear results found."
        
    except Exception as e:
        search_data = f"Error performing web search: {str(e)}"
        
    return {
        "content": [
            {
                "type": "text",
                "text": search_data
            }
        ]
    }



async def chat():
    """Start interactive chat with the research agent."""

    # Check API key first, before creating any files
    ollama_api_key = os.environ.get("OLLAMA_API_KEY")
    if not ollama_api_key:
        print("\nError: OLLAMA_API_KEY not found.")
        print("Set it in your .env file or export it in your shell.")
        print("Create an API key at: https://ollama.com/settings/keys\n")
        return

    # The Claude Agent SDK forwards these variables to its Claude Code process.
    os.environ["ANTHROPIC_BASE_URL"] = OLLAMA_BASE_URL
    os.environ["ANTHROPIC_AUTH_TOKEN"] = ollama_api_key
    os.environ["ANTHROPIC_API_KEY"] = ""

    # Setup session directory and transcript
    transcript_file, session_dir = setup_session()

    # Create transcript writer
    transcript = TranscriptWriter(transcript_file)

    # Load prompts
    lead_agent_prompt = load_prompt("lead_agent.txt")
    story_researcher_prompt = load_prompt("story_researcher.txt")

    # Initialize subagent tracker with transcript writer and session directory
    tracker = SubagentTracker(transcript_writer=transcript, session_dir=session_dir)


    tool_server = create_sdk_mcp_server(
        name="web-tools",
        version="1.0.0",
        tools=[WebSearchCustom]
    )
    
    # Define specialized subagents
    agents = {
         "websearch_agent": AgentDefinition(
                description=(
                    "Use this agent when you need to find content using the local search tool from the web and save to a local file."
                ),
                tools=["Write","WebSearchCustom"],
                #mcpServers=["web"],
                prompt=load_prompt("filewriter_agent.txt"),
                model=OLLAMA_MODEL
            ),
        # "websearch_agent": AgentDefinition(
        #     description=(
        #         "Use this agent when you need to research on any topic from the internet.The content generated will need to be written to a local file"
        #     ),
        #     tools=["Write", "mcp__web__WebSearchCustom"],
        #     mcpServers=["web"],
        #     prompt=load_prompt("websearch_agent.txt"),
        #     model=OLLAMA_MODEL
        # ),
        # "story_teller": AgentDefinition(
        #     description=(
        #         "Use this agent AFTER researchers have completed their work to generate story notes for children "
        #         "The story_teller reads notes from files/story_notes/, creates meaningful end to end stories, which are complete."
        #         "Saves stories to files/stories/ Use this before the story_critic subagent is run on these stories."
        #     ),
        #     tools=["Glob", "Read", "Bash", "Write"],
        #     prompt=story_teller_prompt,
        #     model=OLLAMA_MODEL
        # )
    }

    # Set up hooks for tracking
    hooks = {
        'PreToolUse': [
            HookMatcher(
                matcher=None,  # Match all tools
                hooks=[tracker.pre_tool_use_hook]
            )
        ],
        'PostToolUse': [
            HookMatcher(
                matcher=None,  # Match all tools
                hooks=[tracker.post_tool_use_hook]
            )
        ]
    }

    options = ClaudeAgentOptions(
        permission_mode="bypassPermissions",
        setting_sources=["project"],  # Load skills from project .claude directory
        #system_prompt= "<|think|>\n" + load_prompt("orchestrator_agent.txt"),
        system_prompt= load_prompt("leader_agent1.txt"),
        mcp_servers={"web": tool_server},
        #tools=["Task", "mcp__web__WebSearchCustom", "Write"],
        #tools=["Task"],
        allowed_tools=["Task"],
        #allowed_tools=["Task"],
        #allowed_tools=["mcp__web__WebSearchCustom"],
        #allowed_tools=["WebSearchCustom", "Write"],
        agents=agents,
        hooks=hooks,
        model=OLLAMA_MODEL
    )

    print("\n" + "=" * 50)
    print("  Story Teller Agent")
    print("=" * 50)
    print("\nResearch any topic and get a story in PDF")
    print("\nType 'exit' to quit.\n")

    try:
        async with ClaudeSDKClient(options=options) as client:
            while True:
                # Get input
                try:
                    user_input = input("\nYou: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not user_input or user_input.lower() in ["exit", "quit", "q"]:
                    break

                # Write user input to transcript (file only, not console)
                transcript.write_to_file(f"\nYou: {user_input}\n")

                # Send to agent
                await client.query(prompt=user_input)

                transcript.write("\nAgent: ", end="")

                # Stream and process response
                async for msg in client.receive_response():
                    if type(msg).__name__ == 'AssistantMessage':
                        process_assistant_message(msg, tracker, transcript)

                transcript.write("\n")
    finally:
        transcript.write("\n\nGoodbye!\n")
        transcript.close()
        tracker.close()
        print(f"\nSession logs saved to: {session_dir}")
        print(f"  - Transcript: {transcript_file}")
        print(f"  - Tool calls: {session_dir / 'tool_calls.jsonl'}")


if __name__ == "__main__":
    asyncio.run(chat())
