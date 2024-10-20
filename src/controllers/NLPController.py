
from routes.shemes.givenData import SummaryChat

import logging

class NLPController:
    def __init__(self, DataModelObject, providerObject,user_id: str):
        self.data_model = DataModelObject
        self.provider = providerObject
        self.user_id = user_id
        self.logger = logging.getLogger("uvicorn.error")
    
    async def manage_history(self):
        
        # call the history 
        old_chat_history_summary = await self.data_model.get_chatSummary(user_id = self.user_id)
                    
        
        
        # take the new history
        last_chat_history = self.format_chat_as_string(messages = self.provider.get_chat_state())
        
        # summary the new history
        last_history_summary =  self.provider.summarize_chat(chat = last_chat_history)
        
        
        
        if old_chat_history_summary is not None:
            
            # summary the tow
            new_chat_history_summary =  self.provider.summarize_two_summaries(
                summary1 = old_chat_history_summary,
                summary2 = last_history_summary
            )
                
        else:
            new_chat_history_summary = last_history_summary
            
            
        # update the chat and summary in db
        result = await self.data_model.update_chatSummary(
            data =  SummaryChat(
                user_id = self.user_id,
                summary = new_chat_history_summary
            )
        )
        
        if result:
            self.logger.info("History updated successfully")
        else:
            self.logger.error("Failed to update history")
            
    def format_chat_as_string(self, messages):
        """Formats the output of the method like a string, every line is a message with the speaker and takes care of the order."""
        output_string = ""
        for message in messages['messages']:
            if message.type == 'human':
                speaker = 'Human'
            elif message.type == 'ai':
                speaker = 'AI'
            elif message.type == 'tool':
                speaker = 'Tool'
            else:
                speaker = 'Unknown'
            output_string += f"{speaker}: {message.content}\n"
        return output_string