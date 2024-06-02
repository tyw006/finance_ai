# Import standard libraries
import os
from typing_extensions import TypedDict
from typing import List
# Import embedding library
from langchain_voyageai import VoyageAIEmbeddings
# Import vector db libraries
from langchain_community.vectorstores import Qdrant
import qdrant_client
# Import retrieval libraries
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_groq import ChatGroq
# Import libraries for chains
from langchain.schema import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field 
from langchain_cohere import ChatCohere
# Import web search tool
from langchain_community.tools.tavily_search import TavilySearchResults
# Import LangGraph libraries
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

# API Keys
VOYAGE_API_KEY = os.environ['VOYAGE_API_KEY']
COHERE_API_KEY = os.environ['COHERE_API_KEY']
TAVILY_API_KEY = os.environ['TAVILY_API_KEY']
QDRANT_API_KEY = os.environ['QDRANT_API_KEY']

# Enabling Langsmith Trace
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_PROJECT'] = 'personal finance adaptive RAG'

# Establish embedding model
embeddings = VoyageAIEmbeddings(
    voyage_api_key=VOYAGE_API_KEY,
    model = "voyage-large-2-instruct"
)

# Connect to local docker deployed vector database or cloud URL
cloud_url = "qdrant url here"
client = qdrant_client.QdrantClient(
    cloud_url,
    api_key=QDRANT_API_KEY
)
vectorstore = Qdrant(
    client=client,
    collection_name="reddit personal finance wiki",
    embeddings=embeddings
)

# Establish retriever
retriever = vectorstore.as_retriever( 
    # search_type="mmr",
    search_kwargs={'k':4},
    return_source_documents=True)
# Using Groq for Retrieval filter
groq = ChatGroq(temperature=0,
                model_name="llama3-70b-8192",
                groq_api_key=os.environ['GROQ_API_KEY'])
_filter = LLMChainFilter.from_llm(groq)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=_filter, base_retriever=retriever
)

# Establish LLM
llm = ChatCohere(
    model='command-r-plus',
    temperature=0, 
    ohere_api_key=COHERE_API_KEY
    )

# Establish structured response models
class websearch(BaseModel):
    """
    This tool is used to search the internet for questions that are unrelated to personal finance.
    """
    query: str = Field(description="The query to use when searching the internet.")

class vectorstore(BaseModel):
    """
    A vector store that contains documents relating to personal finance.
    Topics range from emergency funds, student loans, 401K, to paying down debt and buying a home and more.
    """
    query: str = Field(description="The query to use when searching the vectorstore.")

class RetrievalGrader(BaseModel):
    """Checking that the retrieved documents are related to the question/query.
    Score is 'yes' or 'no'."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

class HallucinationGrader(BaseModel):
    """Checking that there are no hallucinations in the generated answer.
    Score is 'yes' or 'no'."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")
class AnswerGrader(BaseModel):
    """Checking that the generated answer addresses the question.
    Score is 'yes' or 'no'."""
    binary_score: str = Field(description="Answer address the question, 'yes' or 'no'")

# Establish web search tool
web_search_tool = TavilySearchResults(api_key=TAVILY_API_KEY)
# Define LangGraph state
class GraphState(TypedDict):
    """ The state of the graph.
    Attributes:
    question: question
    generation: LLM generation
    documents: list of documents"""
    question : str
    generation : str
    documents : List[str]
# Define Routing node
def route_question(state):
    """
    Route question to web search or vector database.
    Args:
        state(dict): Current graph state. Graph state defined in class.
    Returns:
        str: next node to call
    """
    print(">>>ROUTE QUESTION")

    preamble = """You are an expert at routing a user question to a vectorstore or web search.
            The vectorstore contains documents relating to personal finance topics. 
            For questions related to personal finance, use the vectorstore. 
            If you do not know the answer, or the user is requesting for more recent data,
            use web search."""
    llm_router = llm.bind_tools(tools=[websearch, vectorstore], preamble=preamble)

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{question}")
        ]
    )
    question_router = route_prompt | llm_router
    question = state["question"]
    decision = question_router.invoke({"question": question})
    # Fallback to LLM if no decision
    if "tool_calls" not in decision.additional_kwargs:
        print(">>>ROUTE QUESTION TO LLM")
        return "llm_fallback"
    if len(decision.additional_kwargs["tool_calls"]) == 0:
        raise "Router could not decide source."
    # Choose route
    route = decision.additional_kwargs["tool_calls"][0]["function"]["name"]
    if route == 'websearch':
        print(">>>ROUTE QUESTION TO WEB SEARCH")
        return "web_search"
    elif route =='vectorstore':
        print(">>>ROUTE QUESTION TO RAG")
        return "vectorstore"
    else:
        print(">>>ROUTE QUESTION TO LLM")
        return "llm_fallback"
# Define LLM fallback node
def llm_fallback(state):
    """
    Generate answer using the LLM w/o vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print(">>>LLM Fallback")

    preamble = """You are an assistant that graciously answers questions.
                Answer any question presented to you.
                If you don't know the answer, say I don't know."""

    llm_fb = llm.bind(preamble=preamble)

    prompt = lambda x: ChatPromptTemplate.from_messages(
        [
            HumanMessage(
                f"Question: {x['question']} \nAnswer: "
            )
        ]
    )

    fallback_chain = prompt | llm_fb | StrOutputParser()
    question = state["question"]
    generation = fallback_chain.invoke({"question": question})
    return {"question": question, "generation": generation}

# Define web search retrieval node
def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print(">>>WEB SEARCH")

    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n\n".join([d["content"] for d in docs])
    sources = "\n".join([d['url'] for d in docs])
    web_results = Document(page_content=web_results, metadata={'source': sources})

    return {"documents": web_results, "question": question}
# Define vector db retrieval node
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print(">>>RETRIEVE")
    question = state["question"]

    # Retrieval
    documents = compression_retriever.invoke(question)
    # Keep only page content and metadata of documents
    docs_clean = [Document(
    page_content=doc.page_content,
    metadata=doc.metadata) for doc in documents]

    return {"documents": docs_clean, "question": question}
# Define retrieval grader node
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print(">>>CHECK DOCUMENT RELEVANCE TO QUESTION")
    preamble = """
    You are a grader evaluating relevance of the retrieved documents to the user's question.
    If the document contains keyword(s) or a semantic meaning related to the user's question,
    grade it as relevant. The score that should be given is either 'yes' or 'no' to show that 
    the document is relevant to the question.
    """
    structured_llm_grader = llm.with_structured_output(RetrievalGrader, preamble=preamble)

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", """Is the following document related to my question?
            {document}
            Question: {question}""")
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print(">>>GRADE: DOCUMENT RELEVANT")
            filtered_docs.append(d)
        else:
            print(">>>GRADE: DOCUMENT NOT RELEVANT")
            continue
    return {"documents": filtered_docs, "question": question}

# Define decide to generate node
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print(">>>ASSESS GRADED DOCUMENTS")
    question = state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(">>>DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, WEB SEARCH")
        return "web_search"
    else:
        # We have relevant documents, so generate answer
        print(">>>DECISION: GENERATE")
        return "generate"
    
# Define answer generation node
def generate_answer(state):
    """
    Generate answer using the vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print(">>>GENERATE")

    preamble= """You are an an expert financial advisor that uses the following documents to answer questions.
    If you don't know the answer, just say "I don't know."
    Keep the answer as concise as possible, a maximum of three senteces."""

    llm_gen = llm.bind(preamble=preamble)

    prompt = lambda x: ChatPromptTemplate.from_messages(
        [
            HumanMessage(
                f"Question: {x['question']} \nAnswer: ",
                additional_kwargs={"documents": x["documents"]}
            )
        ]
    )

    rag_chain = prompt | llm_gen | StrOutputParser()

    question = state["question"]
    documents = state["documents"]

    if not isinstance(documents, list):
        documents = [documents]

    # RAG generation
    generation = rag_chain.invoke({"documents": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}
# Define hallucination/answer grader node
def hallucination_and_answer_grader(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print(">>>CHECKING HALLUCINATIONS")
    # Define hallucination grading chain
    preamble = """You are an evaluator assessing whether the answer provided is grounded in or supported by retrieved documents.
    You only have to reply in 'yes' or 'no'. 'Yes' means the answer is grounded in and supported by the retrieved documents.
    'No' means the answer is not grounded in the retrieved documents.
    """

    hallucination_grader = llm.with_structured_output(HallucinationGrader, preamble=preamble)

    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", """Retrieved Documents:
            {documents}
            Generated answer: {generation}""")
        ]
    )

    hallucination_chain = hallucination_prompt | hallucination_grader

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_chain.invoke({"documents": documents, "generation": generation})
    
    grade = score.binary_score
    
    # Check hallucination
    if grade == "yes":
        print(">>>DECISION: GENERATION IS GROUNDED IN DOCUMENTS")
        # Check question-answering
        print(">>>GRADE GENERATION vs QUESTION")
        # Define answer grader chain
        preamble1 = """You are a grader that will give a "yes" or "no" score if the LLM generated answer 
        resolves the given question.
        Yes means that the LLM generated answer addresses the question."""

        answer_llm_grader = llm.with_structured_output(AnswerGrader, preamble=preamble1)

        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", """User question:
                {question}
                LLM generated answer: {generation}""")
            ]
        )

        answer_chain = answer_prompt | answer_llm_grader

        score = answer_chain.invoke({"question": question,"generation": generation})
        grade = score.binary_score
        
        if grade == "yes":
            print(">>>DECISION: GENERATION ADDRESSES QUESTION")
            return "useful"
        else:
            print(">>>DECISION: GENERATION DOES NOT ADDRESS QUESTION")
            return "not useful"
    else:
        print(">>>DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, USE WEB SEARCH")
        return "not useful"
    
# Build Graph
workflow = StateGraph(GraphState)

# For chat memory
memory = SqliteSaver.from_conn_string(":memory:")
# Define nodes
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("llm_fallback", llm_fallback)

# Set start point of graph
workflow.set_conditional_entry_point(
    route_question, # function calls what to do next
    {"web_search": "web_search", # mapping output of route_question to node
     "vectorstore": "retrieve",
     "llm_fallback": "llm_fallback"}
)

workflow.add_edge("web_search", "generate_answer") # web_search node routes to "generate"
workflow.add_edge("retrieve", "grade_documents") # retrieving document goes to document grader
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate, # function that outputs web_search again or generate
    {"web_search": "web_search",
     "generate":"generate_answer"
     }
)
workflow.add_conditional_edges(
    "generate_answer",
    hallucination_and_answer_grader,
    {"re-generate": "generate_answer",
     "not useful": "web_search",
     "useful":END
     }
)
workflow.add_edge("llm_fallback", END)

# assemble all nodes and edges
graph = workflow.compile(checkpointer=memory)

