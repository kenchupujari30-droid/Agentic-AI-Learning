
Parts 4 & 5 — Prompt Templates and Structured Output
This notebook covers two closely related ideas: how to build reusable, variable-driven prompts instead of rewriting strings from scratch every time, and how to force a model's reply into a guaranteed, validated shape instead of hoping it phrases things consistently.

By the end of this notebook, you'll be able to build a small pipeline that takes a messy, differently-worded piece of text and reliably turns it into a clean, structured Python object — every single time, regardless of how the input is phrased.

Run each cell yourself, in order.


import os
from dotenv import load_dotenv
load_dotenv()
assert os.environ.get("OPENAI_API_KEY"), "Missing OPENAI_API_KEY"

from langchain.chat_models import init_chat_model
openai_model = init_chat_model("openai:gpt-5-mini")
print("Model ready.")

     
Part 4 — Prompt Templates
4.1 The problem with rebuilding prompts from scratch
If you're building a prompt in code, the natural first instinct is an f-string:

prompt = f"Generate a fun fact about {topic}"
This works for a single, simple case. But the moment you need the same structure with different tones, different roles, or multiple variables, you end up rewriting that string by hand every time — and a small typo can break it silently, with no warning.

A prompt template solves this by separating the fixed structure from the parts that actually change. You define the structure once, and fill in only the blanks each time you use it.

4.2 Building a ChatPromptTemplate
ChatPromptTemplate.from_messages() builds a template out of a list of (role, text) pairs, where the text can contain {variable} placeholders. It behaves like the message lists from Part 3 — but with fill-in-the-blank slots instead of fixed text.


from langchain_core.prompts import ChatPromptTemplate

# {tone} and {topic} are placeholders -- the actual values get filled in at call time,
# not when the template is defined.
fun_fact_prompt = ChatPromptTemplate.from_messages([
    ("system", "You generate a single surprising fun fact. Tone: {tone}."),
    ("human", "Topic: {topic}"),
])

# The | (pipe) operator connects the template's OUTPUT directly into the model's INPUT.
# You'll see this exact pattern -- "feed the output of this into that" -- constantly
# throughout the rest of this course.
chain = fun_fact_prompt | openai_model

# We reuse the SAME template three times, with different variable values each time.
for tone, topic in [
    ("playful and silly", "octopuses"),
    ("deadpan and serious", "octopuses"),
    ("excited like a sports commentator", "the Roman Empire"),
]:
    result = chain.invoke({"tone": tone, "topic": topic})
    print(f"[{tone} / {topic}] -> {result.content}\n")

     
What just happened: the underlying prompt structure never changed across all three calls. Only the two variables did. Notice how differently the two "octopuses" outputs sound despite having an identical topic — that's the tone variable doing real work, in a template you wrote exactly once.

This is the core value of a template: you're not writing three prompts. You're writing one prompt with two blanks, and filling those blanks three different ways.

Part 5 — Structured Output
5.1 Why free text doesn't compose
Let's look directly at the problem structured output solves. Below, we ask the model to extract a customer's name and issue from three real-sounding but differently-phrased messages, using nothing but a plain text instruction.


messages = [
    "Hi, I'm Rina, my order arrived broken and I need a refund ASAP.",
    "hey its dave, package never showed up, kinda annoyed tbh",
    "This is Comfort. My item was fine but delivery took 3 weeks - just flagging it, not urgent.",
]

for msg in messages:
    r = openai_model.invoke(f"Extract the customer's name and issue from: {msg}")
    print(r.content)
    print("---")

     
Look closely at the shape of each answer. Even if all three sound reasonable individually, there's no guarantee they come back in the same format. One might say "Name: Rina, Issue: broken order." Another might phrase it as a full sentence. If you wanted to write code that reads these answers and reliably pulls out just the name, you'd need separate parsing logic for each possible format the model might choose — because the model decides the format fresh every time, not you.

Free text isn't wrong. It's just not a contract. The moment your code depends on a specific shape coming back reliably, you need something stronger than "usually formats it nicely."

5.2 Forcing a guaranteed shape: with_structured_output()
The fix: instead of describing the shape you want in words and hoping the model listens, you define the shape in code using a Pydantic BaseModel, and hand that definition to the model. It cannot return something that doesn't match this shape — if it tries, LangChain catches the mismatch before it ever reaches you.


from pydantic import BaseModel, Field
from typing import Literal

class SupportTicket(BaseModel):
    customer_name: str = Field(description="The customer's first name")
    # Literal restricts this field to EXACTLY these four options -- the model
    # cannot invent a fifth category, Pydantic will reject anything else.
    issue_category: Literal["refund", "delivery", "product_defect", "other"]
    urgency: Literal["low", "medium", "high"]
    needs_human: bool = Field(description="True if this needs a human agent, not a bot")

# with_structured_output() returns a NEW model object that always returns SupportTicket
# instances instead of plain AIMessage objects.
structured_model = openai_model.with_structured_output(SupportTicket)

for msg in messages:
    ticket = structured_model.invoke(f"Extract a support ticket from this message: {msg}")
    print(ticket)

     
Compare this output to section 5.1's output directly. Same three messy input messages, same underlying model — the only thing that changed is that we handed the model a schema first. Every single result now comes back as the exact same SupportTicket shape: name, category, urgency, needs_human. This is already a real Python object you can use directly in code — no string parsing required.

5.3 Pydantic BaseModel vs. TypedDict: two levels of strictness
Pydantic's BaseModel (used above) validates every field strictly — wrong type or wrong shape gets rejected, and LangChain will retry or raise an error. TypedDict defines the same kind of shape but with no validation at all — it's faster and lighter, but nothing stops a malformed result from getting through if the model misbehaves.

When to use which:

Use Pydantic when correctness genuinely matters — production systems, anything where a wrong value has a real consequence.
Use TypedDict when you're prototyping quickly, or you're reasonably confident the model will behave and speed/simplicity matters more.
If you're unsure which to pick, default to Pydantic.
5.4 How the structure actually gets enforced: ToolStrategy vs ProviderStrategy
Under the hood, LangChain has two different mechanisms for forcing a model into a structured shape:

ProviderStrategy uses the model provider's own native JSON-schema output feature. It's fast and reliable, but only works if the specific provider/model combination supports it.
ToolStrategy forces structure by making the model call a synthetic "fake" tool whose arguments match your schema. It works on any model that supports tool calling in general, with slightly more overhead.
If you just pass a schema type directly (as in section 5.2 above), LangChain automatically picks ProviderStrategy when the model supports it, and falls back to ToolStrategy otherwise. You only need to pick one explicitly when you have a specific reason to force it.


from langchain.agents.structured_output import ToolStrategy, ProviderStrategy

tool_strategy_model = openai_model.with_structured_output(ToolStrategy(SupportTicket))
provider_strategy_model = openai_model.with_structured_output(ProviderStrategy(SupportTicket))

test_msg = "Hi, I'm Rina, my order arrived broken and I need a refund ASAP."

print("ToolStrategy result:    ", tool_strategy_model.invoke(test_msg))
print("ProviderStrategy result:", provider_strategy_model.invoke(test_msg))

# For this simple case, both strategies produce the same correct result.
# The difference matters more once structured output is combined with REAL tools
# in the same agent -- covered in Part 11 and later.

     
An honest, important caveat: there are documented, community-reported cases (as of late 2025) where ToolStrategy combined with real tool calls in the same agent can behave unreliably on certain models — hitting recursion limits or failing to execute tool calls correctly. This doesn't mean don't use it; it means test your specific model and use case yourself before trusting this combination for anything important. We revisit and test this exact combination again in Part 11, once real tools are involved.

Full Pipeline: Template In, Structured Output Out
Let's connect both halves of this notebook. A template shapes the question consistently; a structured schema shapes the answer consistently. Chained together, this is a genuinely common production pattern.


extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract a support ticket from the customer's message below."),
    ("human", "{customer_message}"),
])

# Chain the template directly into the STRUCTURED model from section 5.2.
pipeline = extraction_prompt | structured_model

for msg in messages:
    ticket = pipeline.invoke({"customer_message": msg})
    print(ticket)

     
What just happened, end to end: a raw, differently-worded customer message goes into the template, the template's consistently-formatted prompt goes into the structured model, and a validated SupportTicket Python object comes out the other side. No if/else branching to handle different message formats. No manual string parsing. The structure did all the work.

Summary
A prompt template separates fixed structure from variable content, so you write the structure once and reuse it with different inputs.
Free text extraction has no guaranteed shape — the model decides the format fresh every time, which breaks any code that depends on parsing it reliably.
with_structured_output() fixes this by handing the model a schema it must conform to.
Pydantic BaseModel validates strictly; TypedDict is lighter but unvalidated.
ProviderStrategy and ToolStrategy are two different mechanisms for enforcing that schema — auto-selected for you unless you have a reason to force one.
Chaining a template into a structured model with | gives you a complete, predictable pipeline: consistent input framing, consistent output shape.
Next: Part 6 — Tools, where the model starts actually doing things instead of just reasoning and replying.