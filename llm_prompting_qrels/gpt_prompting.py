from openai import OpenAI
from dotenv import load_dotenv
import os
import settings as s

dotenv_path = s.PROJECT_ROOT / '.env'
load_dotenv(dotenv_path)

key = os.environ.get("OPENAI-KEY")
client = OpenAI(
api_key= key
)

def get_llm_response(prompt: str,) -> str:
    completion = client.chat.completions.create(
        model=s.OPEN_AI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        temperature=s.TEMPERATURE,
        top_p=s.TOP_P,
    )
    return completion.choices[0].message.content


