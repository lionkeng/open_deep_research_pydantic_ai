# Overview

Our goal is to recreate a deep-research AI agentic workflow using the exising project [open deep research](https://github.com/langchain-ai/open_deep_research) and implement it with pydantic-ai framework instead of Langgraph.

## Background

We will use the open source project [Open Deep Research](https://github.com/langchain-ai/open_deep_research) from Langgraph as the inspiration. A corresponding blog article is available [here](https://blog.langchain.com/open-deep-research/).

## Mission

We will use the concepts presented in [Open Deep Research](https://github.com/langchain-ai/open_deep_research). But instead of Langgraph we will use [Pydantic-AI](https://ai.pydantic.dev/) as the underlying framework.

### Gain understanding from Pydantic AI documentation

To achieve our mission, explore concepts of multi-agent in Pydantic-AI such as [agent delegation](https://ai.pydantic.dev/multi-agent-applications/#agent-delegation). Explore all concepts about [multi agent](https://ai.pydantic.dev/multi-agent-applications/) including the examples.

### Gain understanding from Pydantic AI github repository

We can also examine Pydantic AI's [code base](https://github.com/pydantic/pydantic-ai) to provide more context for understanding how to implement the application.

### Guidance

1.  We can ignore the [legacy implementation](https://github.com/langchain-ai/open_deep_research/tree/main/src/legacy) in the original repository.

2.  There will be Langgraph or Langchain specific features in the original repository that we will need to translate into equivalent pydantic-ai framework features.

3.  The AI agent will have a web interface that will be implemented with fastAPI. For protocol implementation research on whether the API endpoint should support SSE or websocket.

4.  Implement the interaction between a user and the AI agent in a CLI.

5.  Responses from the AI should be streamed to the user. Consider using

6.  For communication between agents, between AI and user, consider using an Event Bus Architecture for handling async events. The architecture should be lock free. You can see an implementation of this in the [alpine repository](https://github.com/anewgo/alpine), or in the local filesystem in the following directory:

```bash
/Users/keng/work/alpine
```
