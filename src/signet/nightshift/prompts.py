"""LLM prompt templates for autoDream memory consolidation."""

CONSOLIDATION_SYSTEM = """\
You are Signet's memory consolidation system. You are reviewing conversations \
that Signet (an AI research agent) had with users on Discord. Your job is to \
extract durable knowledge from these conversations.

Be precise. Only extract what is genuinely there. Do NOT fabricate, infer \
beyond what was said, or add information that isn't in the messages."""

CONSOLIDATION_PROMPT = """\
Review these conversations and extract the following:

1. **digest**: A 2-4 sentence summary of what was discussed, what conclusions \
were reached, and any open questions or threads left hanging.

2. **entity_facts**: Facts learned about specific people - their preferences, \
expertise, interests, communication style, things they asked about, projects \
they're working on. One entry per fact. Include the person's name.

3. **reflections**: Patterns, recurring themes, or higher-order observations \
that would be worth remembering a week from now. Only genuine insights, not \
restatements of what was said.

Respond with ONLY valid JSON in this exact format:
{{
  "digest": "summary text here",
  "entity_facts": [
    {{"entity": "PersonName", "fact": "what you learned about them"}}
  ],
  "reflections": [
    "observation or pattern"
  ]
}}

If a section has nothing meaningful, use null for digest or empty arrays for \
the others. Do NOT pad with low-quality observations.

---

Messages to process ({message_count} messages, {channel_info}):

{messages}"""
