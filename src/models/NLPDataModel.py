from .BaseDataModel import BaseDataModel
from .enums.db_enums import DataBaseEnum
from routes.shemes import SummaryChat

class NLPDataModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_CHAT_NAME.value]
    # we want to call init_collection
    @staticmethod
    async def create_instance(db_client: object):
        instance = NLPDataModel(db_client)
        await instance.init_collection()
        return instance


    
    # create collection
    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if self.collection not in all_collections:
            self.collection = self.db_client[DataBaseEnum.COLLECTION_CHAT_NAME.value]


    async def update_chatSummary(self, data: SummaryChat):
        result = await self.collection.update_one(
            {'user_id': data.user_id},  # Use 'user_id' as the filter
            {'$set': data.model_dump(by_alias=True, exclude_unset=True)},  # Update the 'summary' field
            upsert=True  # Insert if it doesn't exist
        )
        if result.modified_count > 0 or result.upserted_id is not None:
            return True
        return False

    async def get_chatSummary(self, user_id: str): # get last one from db
        result = await self.collection.find_one({'user_id': user_id})

        if result is None:
            return SummaryChat(user_id=user_id)
        
        return SummaryChat(**result)

    
        