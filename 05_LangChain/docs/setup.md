
import os
from dotenv import load_dotenv
load_dotenv()
True
print(os.environ.get('OPENAI_API_KEY')[0:5])
sk-pr
from langchain.agents import create_agent
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="openai:gpt-5.5",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)
[{'type': 'text', 'text': 'It’s currently sunny in San Francisco.'}]
 