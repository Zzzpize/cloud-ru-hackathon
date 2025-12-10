from models import DbStructure



CHECKLIST_DB: DbStructure = {}

class Database:
    async def _read_db(self) -> DbStructure:
        return CHECKLIST_DB
    
    async def _write_db(self, data: DbStructure) -> None:
        global CHECKLIST_DB
        CHECKLIST_DB = data

db = Database()
