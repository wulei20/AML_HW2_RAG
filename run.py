import os
import logging

from lightrag import LightRAG, QueryParam
from lightrag.llm import zhipu_complete, zhipu_embedding, local_glm4_complete
from lightrag.utils import EmbeddingFunc
from local_model.glm4 import init_glm4

WORKING_DIR = "./working_dir"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

os.environ["ZHIPUAI_API_KEY"] = "2ceb9970b113498bb57d130f6a25abbc.NDc9RQ1sFI5VjOqC"

api_key = os.environ.get("ZHIPUAI_API_KEY")
if api_key is None:
    raise Exception("Please set ZHIPU_API_KEY in your environment")

init_glm4()
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=local_glm4_complete,
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

# with open("./data/questions.txt", 'r', encoding="utf-8") as f:
#     for question in f:
#         rag.query(question, param=QueryParam(mode="local"))
#         rag.query(question, param=QueryParam(mode="global"))
#         rag.query(question, param=QueryParam(mode="hybrid"))

# Perform naive search
print("naive:\n",
    rag.query("What is FRP?", param=QueryParam(mode="naive"))
)

# Perform local search
print("local:\n",
    rag.query("What is FRP?", param=QueryParam(mode="local"))
)

# Perform global search
print("global:\n",
    rag.query("What is FRP?", param=QueryParam(mode="global"))
)

# Perform hybrid search
print("hybrid:\n",
    rag.query("What is FRP?", param=QueryParam(mode="hybrid"))
)
