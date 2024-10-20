# langchine 
from abc import ABC, abstractmethod

from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from .providers.llm_shemes.MassageScheme import State


import logging

class BaseProvider(ABC):
    def __init__(self, model_provider: str):
        # may take the user id and work in different memory chat
        self.model = None
        self.model_provider = model_provider
        self.memory = MemorySaver()
        self.app = None
        
        self.config = {"configurable": {"thread_id": "abc345"}} # needed for langgraph, we don't plan to use it now.
        
        self.logger = logging.getLogger("uvicorn")
    
    @abstractmethod
    def set_model(self, model_name: str): # set the model based on the provider settings
        pass
    
    
    def construct_prompt(self): # you can rewrite this method based on the provider model if you want
        prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You talk like a pirate. Answer all questions to the best of your ability and this the summary of last conversations {summary}.",
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
        
        return prompt
    
    
    async def _call_model(self, state: State):
        if self.model is None:
            self.logger.error(f"{self.model_provider} client was not set")
            return None
        
        print("---" * 50)
        self.logger.info(f"{self.model_provider} client was set in call model")
        
        chain = self.construct_prompt() | self.model
        
        response = await chain.ainvoke(state)
        return {"messages": [response]}
        
        
 
    async def construct_Graph(self):
        
        workflow = StateGraph(state_schema=State)
        workflow.add_edge(START,"start")
        workflow.add_node("start", self._call_model)
        self.app = workflow.compile(checkpointer=self.memory)
    
    async def chat(self, query: str, summary: str):
        
        
        
      #  input_messages = [HumanMessage(query)]
        if self.app is None:
            await self.construct_Graph()
        
        input_massage = {
            "messages" : [HumanMessage(query)],
            "summary" : summary
            
        }
        # Run the chat workflow with the input message
        response = await self.app.ainvoke(input_massage, self.config) # last massage should be HumanMessage()
        
        return response["messages"][-1].content

    def get_chat_state(self): # can pass config if working with multiple users
       return self.app.get_state(self.config)[0]
        
    

    def summarize_chat(self, chat: str) -> str: # can be moved to controller class
        # Create a prompt to guide the summary
        prompt_template = """
        You are an AI tasked with summarizing conversations. 
        Here is the chat: "{chat}"

        Please provide a concise summary of the key points from the conversation.
        """

        prompt = PromptTemplate(input_variables=["chat"], template=prompt_template)
        
        # Generate the summary using the passed model
        prompt_with_chat = prompt.format(chat=chat)
        
        prompt = [HumanMessage(prompt_with_chat)]
        response = self.model.invoke(prompt)
        
        return response.content # AIMassage

    def summarize_two_summaries(self, summary1: str, summary2: str) -> str:
        # Create a prompt to guide the summary
        prompt_template = """
        You are an AI tasked with combining and summarizing two summaries of conversations. 
        Here are the summaries:

        Summary 1: "{summary1}"
        Summary 2: "{summary2}"

        Please provide a concise, combined summary of the key points from both summaries.
        """

        prompt = PromptTemplate(input_variables=["summary1", "summary2"], template=prompt_template)

        # Generate the combined summary using the passed model
        prompt_with_summaries = prompt.format(summary1=summary1, summary2=summary2)
        
        prompt = [HumanMessage(prompt_with_summaries)]
        response = self.model.invoke(prompt_with_summaries)
        
        return response.content
    
   # def update_full_history(self, )
   