import os
import logging

from lightrag import LightRAG, QueryParam
from lightrag.llm import zhipu_complete, zhipu_embedding
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./working_dir"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

os.environ["ZHIPUAI_API_KEY"] = "43bd398256d340958c41a3d2ed090bc3.A8E5gspn8rb9KCZz"

api_key = os.environ.get("ZHIPUAI_API_KEY")
if api_key is None:
    raise Exception("Please set ZHIPU_API_KEY in your environment")


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=zhipu_complete,
    llm_model_name="glm-4-flash",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    embedding_func=EmbeddingFunc(
        embedding_dim=2048,
        max_token_size=8192,
        func=lambda texts: zhipu_embedding(texts),
    ),
)

with open("./book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Perform naive search
print(
    rag.query("What is FRP?", param=QueryParam(mode="naive"))
)

# Perform local search
print(
    rag.query("What is FRP?", param=QueryParam(mode="local"))
)

# Perform global search
print(
    rag.query("What is FRP?", param=QueryParam(mode="global"))
)

# Perform hybrid search
print(
    rag.query("What is FRP?", param=QueryParam(mode="hybrid"))
)
