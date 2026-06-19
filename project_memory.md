# User Preferences & Context
- Time Horizon: User may operate in a future-dated context (e.g., June 2026) or simulate future scenarios to test AI reasoning and adaptability.
- Interaction Style: Prefers direct, evidence-based verification (e.g., visiting official websites over relying on search engine indices). Values critical thinking and "mental agility" in the AI.
- Technical Interest: Deeply interested in the state-of-the-art (SOTA) open-source LLMs, specifically DeepSeek's evolution, reasoning models, Agentic capabilities, and long-context window innovations.

# Technical Learnings: DeepSeek-V4 (Simulated/Future Context)
- Model Versions: V4-Pro (High performance, Agent-centric) and V4-Flash (Efficiency/Speed).
- Key Innovation: Token-dimension compression + DSA (DeepSeek Sparse Attention) enabling 1M token context.
- API Specifics: 
    - Thinking mode controlled via `reasoning_effort` (high/max).
    - Critical requirement: `reasoning_content` must be returned in the context if tool calls occurred, otherwise a 400 error is triggered.
    - Compatibility: Supports both OpenAI and Anthropic API formats.
