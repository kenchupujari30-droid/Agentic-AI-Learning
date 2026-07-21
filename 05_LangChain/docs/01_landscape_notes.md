
Part 1 — Understanding the LangChain Ecosystem
Welcome to the first module of this course. Before writing a single line of code, it's worth spending real time understanding what LangChain actually is, how it relates to the other tools in its family (LangGraph, LangSmith, Deep Agents), and why the framework looks the way it does today. Skipping this step is the single biggest reason people get confused later — they start writing code without a mental map of where that code fits.

This notebook has no code to run yet. Think of it as the map you'll keep referring back to for the rest of the course.

1. The "Lang" Family: Four Products, One Team, Four Different Jobs
If you've searched anything about LangChain online, you've probably already run into blog posts titled something like "LangChain vs LangGraph vs LangSmith — Which One Should I Use?" This is one of the most common points of confusion for beginners, and it exists for a simple reason: the LangChain team builds four different products that share the same first word, but solve four completely different problems.

None of these four tools are competitors to each other. You don't "pick one." You use as many of them as your project actually needs.

LangGraph — the foundation
LangGraph is a low-level orchestration framework. It's the engine underneath everything else in this family. It gives you fine-grained control over how an application flows: branching logic, loops, persistence (remembering state between steps), and durable execution (being able to pause and resume a long-running process).

Most people building agents will never touch LangGraph directly, at least not at first — because LangChain (the next product) is built specifically to hide that complexity until you actually need it.

LangChain — what this entire course is built on
LangChain provides create_agent: a highly configurable harness for building agents quickly. It's built on top of LangGraph, which means you get LangGraph's reliability (durable execution, persistence, human-in-the-loop support) without needing to understand LangGraph's internals.

Official docs quote, word for word:

"LangChain's agents are built on top of LangGraph. This allows us to take advantage of LangGraph's durable execution, human-in-the-loop support, persistence, and more."

This is the product this entire course is built around, starting from Part 2 onward.

Deep Agents — batteries included
Deep Agents is a layer built on top of LangChain's create_agent. Where create_agent gives you a highly customizable but empty harness, Deep Agents gives you a harness that already comes with planning tools, a virtual filesystem, and the ability to spawn sub-agents — all pre-wired.

You reach for Deep Agents when you want to skip assembling those capabilities yourself. We'll build with Deep Agents properly in a later module of this course; for now, just know it's the third tier in this family, not a separate, unrelated tool.

LangSmith — the one that's fundamentally different
LangSmith is not something you "build an agent with." It's an observability and evaluation platform — it watches agents built with any of the other three tools (and, since it supports OpenTelemetry, technically almost anything else too) and lets you see exactly what happened during a run.

Here's the core problem LangSmith solves. With a normal Python script, if something goes wrong, you read the stack trace and the code, and you can usually figure out exactly what happened. An agent is different — it decides what to do at runtime, based on a conversation it has with itself (calling tools, reasoning about results, calling more tools). You cannot "read the code" to know what an agent actually did on a specific run, because the code only describes what it could do, not what it did do on that particular call.

LangSmith solves this by recording a trace — a complete, timestamped record of every model call, every tool call, and every decision the agent made, in order. When something goes wrong, you don't guess. You open the trace and read exactly what happened, step by step.

Two environment variables turn this on:

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key-here
That's the entire setup cost. We'll actually turn this on and look at a real trace in a later module — for now, understand why it exists: an agent's code tells you what it's capable of; a trace tells you what it actually did.

A note on cost: LangSmith's free Developer tier includes 5,000 traces per month as of 2026, which is more than enough for learning and personal projects. You'd only need to think about paid tiers once you're running something in production with real user traffic.

2. The Single Most Important Sentence in This Course
The official LangChain documentation opens with this exact framing, and it's worth memorizing:

An agent is a model calling tools in a loop until a given task is complete. A harness is everything around that loop.

Let's unpack both halves of that.

The model is the raw language model — GPT, Claude, Gemini, whatever you plug in. It is extraordinarily capable at reasoning, but on its own it has no ability to actually do anything. It doesn't know what tools exist. It has no instructions on how to behave in your specific application. It can't check anything outside of what it already knows.

The harness is everything that turns that raw capability into something useful:

The system prompt — instructions on how the agent should behave
The tools — what it's actually allowed to reach for and use
The middleware — checkpoints that shape its behavior at every step (we cover this in depth starting in Part 8)
create_agent is that harness. When you call create_agent(model=..., tools=..., system_prompt=...), you are configuring the harness around a model — not building a new model.

Why this distinction matters in practice: almost everything you'll learn for the rest of this course — retrieval-augmented generation, memory, multi-agent systems, database agents — is just a different configuration of the harness around the same underlying model. The model doesn't change. What changes is what you give it access to, and what rules govern how it uses that access.

The three tiers, precisely
LangGraph        →  Build the harness from raw parts yourself. Maximum control, most effort.
LangChain         →  create_agent gives you a pre-configured, highly customizable harness.
                      This is what the rest of this course uses.
Deep Agents       →  create_deep_agent gives you a harness that already includes planning,
                      a filesystem, and sub-agent spawning, pre-wired.
All three are built on the same underlying LangGraph engine. You are not "avoiding" LangGraph by using LangChain — you're standing on it, you just don't need to look at it directly for most of this course.

3. A Real Timeline — Not "It's Changed a Lot"
If you search for LangChain tutorials online, you'll find content spanning several years, and a huge amount of it is now outdated — not because it was wrong when it was written, but because the framework has gone through a genuine architectural overhaul. Knowing the actual timeline lets you judge whether something you're reading is current.

Date	What happened
Oct 2022	LangChain launches. Two core ideas: LLM abstractions, and "Chains" — predetermined sequences of computation (e.g. a RAG chain: retrieve documents, then generate an answer).
Dec 2022	The first general-purpose agents are added, based on the ReAct paper (Reasoning + Acting). The model generates JSON representing a tool call, and LangChain parses that JSON by hand.
Feb 2024	LangGraph is released — the low-level orchestration layer that had been missing. It adds streaming, durable execution, short-term memory, and human-in-the-loop support.
Oct 2024	LangGraph becomes the preferred way to build anything beyond a single model call. Most of the old LangChain chains and agents are marked deprecated.
Oct 20, 2025	LangChain v1.0 ships. One unified agent abstraction, built on top of LangGraph, replaces essentially everything that came before. Code using the old approach still runs, but only if you deliberately install a separate langchain-classic package.
Mar 15, 2026	Deep Agents is released — the batteries-included harness described above.
The practical takeaway: for two years (2022–2024), building an agent meant hand-parsing JSON into tool calls — this is what classes like AgentExecutor and functions like initialize_agent do. That approach didn't scale well, which is exactly why LangGraph was built. Then, in October 2025, everything was unified into the single create_agent abstraction this course teaches.

If you ever see a 2026-dated tutorial using AgentExecutor, it is almost certainly outdated — not wrong at the time it was written, just written before the ground moved. If you try to run old code that imports AgentExecutor directly from langchain, you'll get an ImportError — this isn't a bug in your setup, it's because that functionality now deliberately lives in the separate langchain-classic package, kept apart so the main v1.0 package stays clean.

4. Quick Self-Check
Before moving to Part 2, you should be able to answer these without looking back:

If someone asks you to build an agent quickly with a lot of customization, which of the four Lang-family tools do you reach for?
If your agent is misbehaving in production and you need to know exactly what happened on one specific run, which tool do you reach for?
Fill in the blank: "An agent is a model calling ___ in a ___ until a given task is complete."
Why would a tutorial using AgentExecutor be a signal that the content might be outdated?
(Answers: 1 — LangChain's create_agent, or Deep Agents if you want batteries-included. 2 — LangSmith. 3 — "tools", "loop". 4 — AgentExecutor was the pre-v1.0 approach; anything using it was written before October 2025's unification into create_agent.)

Next: Part 2 — Environment Setup, where we actually start writing code.