import os
import argparse

from dotenv import load_dotenv
from openai import OpenAI

from prompts import system_prompt
from call_function import available_functions, call_function


def generate_content(client, messages):
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
        tools=available_functions,
        temperature=0,
    )

    if response.usage is None:
        raise RuntimeError("Response usage data is missing")

    return response


def main():
    parser = argparse.ArgumentParser(description="Chatbot")

    parser.add_argument(
        "user_prompt",
        type=str,
        help="User prompt"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    load_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY")

    if api_key is None:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": args.user_prompt,
        },
    ]

    for _ in range(20):
        response = generate_content(client, messages)

        message = response.choices[0].message

        messages.append(message)

        if message.tool_calls:
            for tool_call in message.tool_calls:
                result_message = call_function(
                    tool_call,
                    verbose=args.verbose,
                )

                if not result_message["content"]:
                    raise RuntimeError(
                        "Tool call returned empty content"
                    )

                messages.append(result_message)

                if args.verbose:
                    print(
                        f"-> {result_message['content']}"
                    )
        else:
            print("Final response:")
            print(message.content)
            return

    print("Error: Maximum iterations reached")


if __name__ == "__main__":
    main()
