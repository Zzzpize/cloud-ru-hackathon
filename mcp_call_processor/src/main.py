import os
from dotenv import load_dotenv

load_dotenv()

from mcp_instance import mcp
import tools 

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))

    mcp.run(
        transport="streamable-http", 
        host="0.0.0.0", 
        port=port, 
        stateless_http=True
    )