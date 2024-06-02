import streamlit as st
from langserve import RemoteRunnable

# Set page configs
st.set_page_config(
    page_title="Financial AIdvisor",
    page_icon= ":robot_face:"
)

graph_chain = RemoteRunnable("http://localhost:8000/chat")
thread_id = {"thread_id": "1"} # for graph memory

first_msg = """Hello! I am your financial overlord. Ask me any question 
and I'll do my best to find the answer for you."""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": first_msg, "source": ""}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if query := st.chat_input("Enter your question here:..."):
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.spinner("Fetching your answer..."):
        with st.chat_message("ai"):
            # message_placeholder = st.empty()
            response = graph_chain.invoke({"question" : query, "thread_id": thread_id})
            st.write(response['answer'])
            st.write(response['sources'])
    st.session_state['messages'].append({"role": "ai",
                                         "content": response['answer'] + '\n' + response['sources']})