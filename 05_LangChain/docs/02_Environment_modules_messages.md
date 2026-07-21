
Parts 2 & 3 — Environment Setup, and Understanding Models and Messages
This notebook covers two things: getting a real LangChain project running on your machine with your API key stored safely, and then understanding exactly what you get back every time you call a model — not just the text reply, but everything else sitting on that response object that most tutorials never show you.

Run every cell in this notebook yourself, in order. Read the explanation before each code cell — it tells you why the code is written the way it is, not just what it does.

Part 2 — Environment Setup
2.1 Installing the packages
Every notebook in this course assumes a working LangChain environment already exists. We set that up once here.

Open a terminal (not this notebook) and run:

uv init langchain-course
cd langchain-course
uv add langchain langchain-openai langchain-community langgraph python-dotenv
uv add langchain-mcp-adapters langchain-chroma chromadb pypdf
uv is a fast Python package manager. uv init creates a new project folder with its own isolated dependencies, and uv add installs packages into it. The second uv add line installs a few packages we won't need until later modules (for connecting to external tools, and for building retrieval systems) — installing them now means you won't need to stop mid-course to add them later.

2.2 Where your API key actually lives
This is the part that's easy to skip and genuinely important not to. Your API key should never be typed directly into a code file that could end up in a public GitHub repository. Instead, it lives in a separate file called .env, which your code reads at runtime but which is explicitly excluded from version control.

Create a file called .env in your project folder:

OPENAI_API_KEY=sk-...
Then confirm your .gitignore file (also in the project folder) contains this line:

.env
Why this matters: if .env isn't listed in .gitignore, the very first time you commit and push your code, your real API key gets uploaded to GitHub along with it — where it can be found and used by anyone, often within minutes, by automated scanners that specifically look for leaked keys. This is one of the most common and most avoidable mistakes when starting out with any API-based project, not just LangChain.


# This cell loads your .env file and confirms your key is available.
# Run this cell at the start of every notebook in this course -- it's your sanity check
# that the environment is set up correctly before anything else runs.

import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file in the current directory and loads its contents
                # into the environment variables Python can see

# This assertion will stop the notebook here with a clear error message if the key
# is missing, rather than letting you hit a confusing authentication error later.
assert os.environ.get("OPENAI_API_KEY"), "Missing OPENAI_API_KEY -- check your .env file"
print("Environment OK.")

     
2.3 Proving it all works: the official quickstart
Rather than inventing a custom "hello world" example, here is the exact quickstart snippet from the official LangChain documentation, unmodified. If this cell runs successfully, your environment is genuinely correct — not just correct enough to pass a simplified example.

Don't worry about understanding every piece of this code right now. create_agent, tool binding, and system prompts are all covered properly starting in Part 6 and Part 7. For now, just confirm it runs.


from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    # This is a placeholder tool -- it doesn't call a real weather API,
    # it just returns a fixed string so we can prove the whole pipeline works.
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="gpt-5-mini",   # This string identifies which model to use.
                            # If this specific model name has been retired or replaced
                            # by the time you're running this, check docs.langchain.com
                            # for the current model catalog and swap this string --
                            # nothing else in this code needs to change.
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)

# .content_blocks is the standardized way LangChain exposes a message's content --
# we explain exactly what this is and why it exists later in this notebook.
print(result["messages"][-1].content_blocks)

     
If this cell throws an authentication error: the two most common causes are (1) your .env file isn't in the same folder you're running this notebook from, or (2) you created .env after starting your notebook session — restart the kernel/runtime so load_dotenv() picks up the file that now exists.

Part 3 — Understanding Models and Messages
3.1 The standard model interface
LangChain provides one consistent way to talk to a model, no matter which company built it. You call init_chat_model() with a string in the format "provider:model", and everything after that — invoking it, streaming it, binding tools to it — works identically regardless of the provider.


from langchain.chat_models import init_chat_model

# The string before the colon is the PROVIDER. The string after is the specific MODEL.
openai_model = init_chat_model("openai:gpt-5-mini")

response = openai_model.invoke("In one sentence, what is LangChain?")
print(response.content)

     
Why this matters: if you later decide to switch from OpenAI to Anthropic's Claude, or to Google's Gemini, or even to a completely free model running on your own machine (we'll do exactly this in section 3.9), you change one string. Nothing else about your code — how you call .invoke(), how you bind tools, how you read the response — needs to change. This is what "a standard model interface" means in practice, not just in theory.

3.2 Messages: System, Human, AI, and Tool
You could, in theory, send a model one long string containing your instructions and the user's question mixed together. This is a bad idea, because the model has no reliable way to tell which part of that string is an instruction and which part is what the user actually said.

The fix is a message list — a sequence of objects, where each one is explicitly labeled with who "said" it:

SystemMessage — instructions for how the model should behave. The model treats this differently from user input; it's the "director's note," not something spoken in the scene.
HumanMessage — what the user actually said or asked.
AIMessage — what the model said in a previous turn (we cover this object in depth below, it's the richest of the four).
ToolMessage — the result of a tool call, sent back to the model so it can continue reasoning with that new information (covered fully in Part 6).

from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="You are a pirate. Answer everything in pirate speak."),
    HumanMessage(content="What's the capital of France?"),
]

response = openai_model.invoke(messages)
print(response.content)

# Notice: the instruction to "be a pirate" was never mixed into the actual question.
# It lives entirely in the SystemMessage, completely separate from what the user asked.
# This separation is the entire point of using a message list instead of one raw string.

     
3.3 What you actually get back: the full AIMessage object
This is the section most tutorials skip entirely, and it's genuinely useful to know. Every time you call a model, you get back an AIMessage object. Printing .content shows you the text reply — but that's only one of several fields sitting on that same object. Let's look at all of them.

Field	What it holds
.text	The plain text reply, guaranteed to always be a string
.content	The raw content — usually the same as .text, but can be a list of content blocks for richer responses
.content_blocks	The standardized structure LangChain uses to represent rich content (text, reasoning, citations, images) consistently across providers
.tool_calls	A list of any tool calls the model requested — empty if it didn't request any
.id	A unique identifier for this specific message
.usage_metadata	Real token counts for this call — input tokens, output tokens, total
.response_metadata	Provider-specific extra information, such as which exact model responded and why it stopped generating

response = openai_model.invoke("Explain agentic AI in one sentence.")

print("text:             ", response.text)
print("content_blocks:   ", response.content_blocks)
print("id:               ", response.id)
print("tool_calls:       ", response.tool_calls)

# tool_calls is an empty list here because we haven't given this model any tools to call.
# We'll revisit this exact field once tools are introduced in Part 6 and Part 7 --
# when a model DOES decide to call a tool, the request shows up right here.

     
3.4 Token usage: cost information you don't have to calculate yourself
You do not need a separate token-counting library to know how much a call cost. The exact numbers the provider billed you for are already sitting on the response, in .usage_metadata.


response = openai_model.invoke("Hello!")
print(response.usage_metadata)

# This typically includes:
#   input_tokens   -- tokens in what you sent
#   output_tokens  -- tokens in what came back
#   total_tokens   -- the sum
#   input_token_details / output_token_details -- finer breakdowns (e.g. cached tokens, reasoning tokens)
#
# If you're building anything that needs to track spend per user or per session, sum this
# field across a conversation -- it's the exact billed number, not an estimate.

     
3.5 Manually inserting an AIMessage into history
Sometimes you need to restore a previously saved conversation, or set up a specific scenario for testing, without actually calling the model for that particular turn. You can construct an AIMessage yourself and place it directly into the message list — the model will treat it exactly as if it had said that itself.


from langchain_core.messages import AIMessage

ai_msg = AIMessage("I'd be happy to help you with that question!")

messages = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Can you help me?"),
    ai_msg,  # inserted directly -- the model never actually generated this,
             # but it will treat it as real prior context
    HumanMessage("Great! What's 2+2?"),
]

response = openai_model.invoke(messages)
print(response.content)

     
Important distinction: this technique is legitimate for restoring genuine conversation history. It is not a way to make the model "believe" it agreed to something for the current turn in a way that biases its behavior dishonestly — because the model treats a fabricated AIMessage as real context either way, misusing this pattern can genuinely mislead the model's later reasoning. Use it for real history reconstruction, not to manufacture fake agreement.

3.6 Streaming: why you get AIMessageChunk objects, and how they combine
When you call .invoke(), you wait for the complete response and get back one AIMessage. When you call .stream() instead, you get the response piece by piece, as it's generated — and each piece is a distinct object called an AIMessageChunk, not a plain string fragment.

The genuinely useful part: these chunks can be added together with the + operator, and the result is a fully reconstructed message, identical in shape to what .invoke() would have returned.


chunks = []
full_message = None

for chunk in openai_model.stream("Write one short sentence about the ocean."):
    chunks.append(chunk)
    print(repr(chunk.text), " <- chunk type:", type(chunk).__name__)
    # Each chunk is summed into the running total using +.
    # On the first chunk, full_message is None, so we just take the chunk directly.
    full_message = chunk if full_message is None else full_message + chunk

print()
print("Number of chunks received:", len(chunks))
print("Reconstructed full message:", full_message.content)
print("Reconstructed type:", type(full_message).__name__)

# Notice: we never called .invoke() here. The full message was built purely by summing
# chunks together, and it comes out identical in shape to what .invoke() would have given us.

     
Why streaming matters in practice: it's what powers real-time chat interfaces, where text appears to the user as it's generated rather than all at once after a delay. We build this directly into agents in Part 7.

3.7 Tool call requests live inside the AIMessage
This is a short preview forward to Part 6 and Part 7. When a model decides it needs to call a tool, that request doesn't live in some separate object — it's sitting in the exact same .tool_calls field we looked at in section 3.3.


def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"Sunny in {location}"

# .bind_tools() makes the model AWARE that this tool exists and can be requested.
# It does NOT execute anything -- that's a separate step, covered in Part 6 and Part 7.
model_with_tools = openai_model.bind_tools([get_weather])

response = model_with_tools.invoke("What's the weather in Paris?")

for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}  Args: {tool_call['args']}  ID: {tool_call['id']}")

# Nothing has actually run yet -- this is just the model's REQUEST to call the tool,
# sitting in the same tool_calls field we saw was empty back in section 3.3.

     
3.8 ToolMessage and the artifact field: two audiences for one tool result
A ToolMessage (the result of running a tool, sent back to the model) has a .content field — this is what the model reads. It can also carry an .artifact field: extra data your application can use, that is never sent to the model.

This matters when a tool needs to hand back more information than the model actually needs to see — for example, a retrieval tool might return the readable passage as .content (for the model), while attaching a document ID or page number as .artifact (for your application to build a citation link with, later).


from langchain_core.messages import ToolMessage

tool_message = ToolMessage(
    content="It was the best of times, it was the worst of times.",
    tool_call_id="call_123",
    name="search_books",
    artifact={"document_id": "doc_123", "page": 0},
)

print("What the MODEL sees:", tool_message.content)
print("What your APP can use (the model never sees this):", tool_message.artifact)

     
3.9 Paid vs. free models: proving the interface really doesn't change
Everything in this notebook has used OpenAI's API, which costs money per call (a small amount for the examples here, but real cost nonetheless). LangChain's standard interface means you can swap to a completely free, locally-run model with a one-string change and nothing else.


# PAID: OpenAI, Anthropic, Google -- all called the exact same way through init_chat_model
paid_model = init_chat_model("openai:gpt-5-mini")

# FREE / LOCAL: Ollama runs entirely on your own machine, no API key, no per-call cost.
# To use this, first install Ollama (ollama.com) and run: ollama pull llama3.2
# Then uncomment the two lines below:

# free_model = init_chat_model("ollama:llama3.2")
# response = free_model.invoke("In one sentence, what is LangChain?")
# print(response.content)

print("Same init_chat_model() call, same .invoke() interface, same message objects returned --")
print("only the provider string changes. That's the entire point of a standard model interface.")

     
A practical note on free/local models: they're genuinely free of API cost, but not free of hardware cost — a model like llama3.2 needs real RAM (and ideally a GPU) to run at a usable speed. Expect noticeably different response quality compared to a frontier paid model too; the interface is identical, the underlying intelligence is not. Free/local models are excellent for development, testing, and situations with no internet access or a tight budget; make an informed, deliberate choice about model quality for anything user-facing.

Summary: What a Model Call Actually Gives You
Putting it all together — every time you call a model in LangChain, you get back an object with:

.text / .content — the reply itself
.content_blocks — a standardized structure for rich content
.tool_calls — any tool requests, ready for Part 6/7
.usage_metadata — real, exact token costs
.id and .response_metadata — traceability and provider-specific detail
And if you stream instead of waiting for the full response, you get a sequence of AIMessageChunk objects that sum together with + into that exact same shape.

None of this required a single extra import or a third-party cost-tracking library. It's all already sitting on the object LangChain hands you back — the skill is knowing to look.

Next: Parts 4 & 5 — Prompt Templates and Structured Output, where we start making the model's input and output shapes fully predictable.