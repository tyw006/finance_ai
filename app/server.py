from fastapi import FastAPI
from langchain_core.runnables import chain
from backend import graph
from langserve import add_routes

app = FastAPI(
    title="Financial AIdvisor",
    version="1.0",
    description="Server for Langgraph AIdvisor"
)

#Custom chain to run langgraph
@chain
def graph_chain(query: dict):
    response = graph.invoke({"question": query['question']},
                        {"configurable": query["thread_id"]})
    answer = response['generation']
    sources = []
    try:
        [sources.append(doc.metadata['source']) for doc in response['documents']
         if doc.metadata['source'] not in sources]
    except KeyError:
        sources = ''
    sources_ = "\n".join(sources)
    return {"answer": answer, "sources": sources_}

add_routes(app,
           graph_chain,
           path="/chat")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
