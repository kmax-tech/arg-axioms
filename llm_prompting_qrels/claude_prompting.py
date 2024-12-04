import anthropic
import os
import settings as s
from dotenv import load_dotenv

dotenv_path = s.PROJECT_ROOT / '.env'
load_dotenv(dotenv_path)

key = os.environ.get("ANTHROPIC_API_KEY2")
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=key,
)

def get_llm_response(prompt):
    #raise Exception("This function is not implemented yet. Please use the 'get_llm_response' function from the 'gemini_prompting' module.")
    message = client.messages.create(
        model=s.CLAUDE_MODEL,
        max_tokens=1000,
        temperature=s.TEMPERATURE,
        top_p = s.TOP_P,

        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    return message.content[0].text