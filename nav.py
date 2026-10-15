import asyncio
from pathlib import Path
from browser_use.llm import ChatOllama 
from browser_use import Agent

# 1. Safely read the task instructions from external text file
steps_file_path = Path("test_steps.txt")

try:
    with open(steps_file_path, "r", encoding="utf-8") as file:
        dynamic_task = file.read()
except FileNotFoundError:
    print(f"❌ Error: The file '{steps_file_path}' was not found. Please create it first.")
    exit(1)

# 2. Initialize local open-source model via Ollama
llm = ChatOllama(
    model="qwen3.5:9b",
    ollama_options={"num_ctx": 32000}
)

# 3. Define the autonomous test objective 
agent = Agent(
    task=dynamic_task,
    llm=llm
)

async def run_nav_test():
    result = await agent.run()
    print("Test Execution Complete:", result)

if __name__ == '__main__':
    asyncio.run(run_nav_test())