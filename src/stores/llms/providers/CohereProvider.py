
from ..BaseProvider import BaseProvider

from langchain_cohere import ChatCohere


class CohereProvider(BaseProvider):
    def __init__(self, api_key: str,
                 temperature: float=0.2):
        
        super().__init__("Cohere")
        
        self.temperature = temperature
        self.api_key = api_key
        
        
        
    
    def set_model(self, model_name: str):
        self.model = ChatCohere(model=model_name,
                                cohere_api_key=self.api_key,
                                temperature=self.temperature
                                )
        self.logger.info(f"{self.model_provider} model was set")
        
 
        