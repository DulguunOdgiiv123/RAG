"""
Phase 1: Claude API basics — multi-turn conversation loop.

Demonstrates:
- Manual conversation history (the API is stateless; you resend
  the full message list every turn)
- Reading response content by block type, not position (adaptive
  thinking on newer models can insert a ThinkingBlock before the
  text block)
- Token usage growing turn over turn as history accumulates
- system prompt for persistent behavior/persona across turns

Note: claude-sonnet-5 has deprecated manual `temperature` control
(non-default values return a 400 error) in favor of adaptive
thinking. Use system-prompt instructions to steer tone/style
instead. Older/smaller models (e.g. claude-haiku-4-5-20251001)
still accept temperature.
"""

from dotenv import load_dotenv
load_dotenv()

import anthropic

client = anthropic.Anthropic()

SYSTEM_PROMPT = "You are a sarcastic pirate. Answer everything in pirate speak, and be a bit sarcastic."


def get_reply_text(response):
    """Extract the text block from a response's content list.
    Never assume content[0] is text — thinking blocks can come first."""
    for block in response.content:
        if block.type == "text":
            return block.text
    return None


def chat():
    messages = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        messages.append({"role": "user", "content": user_input})

        response = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=messages
        )

        reply_text = get_reply_text(response)
        print(f"Claude: {reply_text}")
        print(f"[tokens — input: {response.usage.input_tokens}, output: {response.usage.output_tokens}]")

        messages.append({"role": "assistant", "content": reply_text})


if __name__ == "__main__":
    chat()
