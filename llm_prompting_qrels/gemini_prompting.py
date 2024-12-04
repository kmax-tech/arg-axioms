import google.api_core.exceptions
from google.cloud import aiplatform
from google.oauth2 import service_account
from google.cloud import aiplatform_v1
import os
import vertexai
from vertexai.generative_models import GenerationConfig, Part
from vertexai.generative_models import GenerativeModel, ChatSession
import settings as s
from loguru import logger
import random
def get_llm_response(prompt: str) -> str:
    mycredentials = service_account.Credentials.from_service_account_file(s.GEMINI_CRED)
    vertexai.init(credentials=mycredentials , project='argu-walton-class-gen' , location='europe-west4')
    locations = ['us-central1' , 'us-east1' , 'us-east4' , 'europe-west1' , 'europe-west4']
    index_ref = random.randint(0, len(locations) - 1)
    new_location = locations[index_ref]
    while True:
        try:
            vertexai.init(credentials=mycredentials,project='argu-walton-class-gen', location=new_location)
            config = GenerationConfig(temperature=s.TEMPERATURE , top_p=s.TOP_P)
            model = GenerativeModel(s.GEMINI_MODEL)
            chat = model.start_chat()  # indicate start of the chat session
            responses = chat.send_message(prompt , stream=False ,
                                          generation_config=config)
            return responses.text

        except google.api_core.exceptions.ResourceExhausted as e:
            logger.error(f"Resource exhausted error: {e}")
            new_location = locations[index_ref]
            logger.info(f"Switching to location {new_location}")
            index_ref += 1
            if index_ref == len(locations):
                index_ref = 0


