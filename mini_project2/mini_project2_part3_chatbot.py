# Python
# Import the necessary libraries
from pinecone import Pinecone
import json
import sys
import streamlit as st
from openai import OpenAI

# ====================== Agent Implementations ======================
# Template for Obnoxious, Relevance and Prompt Injection Agents.
class Filtering_Agent:
    def __init__(self, client, model: str) -> None:
        # TODO: Initialize the client and prompt for the Filtering_Agent
        pass

    def set_prompt(self, prompt):
        # TODO: Set the prompt for the Filtering_Agent
        pass
    def extract_action(self, response) -> bool:
        # TODO: Extract the action from the response
        pass
    def check_query(self, query):
        # TODO: Check if the query is obnoxious or not
        return dict({
            "obnoxious": False, "relevant": True, "prompt_injection": False, "is_greetings": False
        })

class Query_Agent:
    def __init__(self, pinecone_index, openai_client, embedding_model: str) -> None:
        self.client = openai_client
        self.index = pinecone_index
        self.model = embedding_model

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding

    def query_vector_store(self, query: str, k=5):
        query_vector = self.get_embedding(query)  # Get the embedding vector of the query

        results = self.index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True,
            namespace="ns2500"
        )
        return results

    def set_prompt(self, prompt):
        # TODO: Set the prompt for the Query_Agent agent
        pass

    def extract_action(self, response, query = None):
        # TODO: Extract the action from the response
        pass

class Answering_Agent:
    def __init__(self, openai_client, model: str) -> None:
        # TODO: Initialize the Answering_Agent
        self.client = openai_client
        self.model = model

    def generate_response(self, query, docs, conv_history, mode, k=5):
        # TODO: Generate a response to the user's query
        context = "\n".join([d["metadata"]["text"] for d in docs["matches"]])
        history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conv_history[-3:]])

        mode_instruction = {
            "chatty": "Respond in a friendly, verbose manner",
            "concise": "Respond in a professional, concise manner",
            "funny": "Add humor to your responses"
        }.get(mode, "")

        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context and history.
        Note: It is forbidden to use your own knowledge to answer these questions!

        Context: {context}
        History: {history}
        Instruction: {mode_instruction}
        Query: {query}"""

        return self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

class Head_Agent:
    def __init__(self, openai_key, pinecone_key, pinecone_index_name, model) -> None:
        self.model = model
        self.client = OpenAI(api_key=openai_key)
        self.index = Pinecone(api_key=pinecone_key).Index(pinecone_index_name)
        self.setup_sub_agents()

    def setup_sub_agents(self):
        self.filtering_agent = Filtering_Agent(self.client, self.model)
        self.query_agent = Query_Agent(self.index, self.client, "text-embedding-3-small")
        self.answering_agent = Answering_Agent(self.client, self.model)

    def main_loop(self, query, mode, conv_history):
        # Check query validity
        filter_result = self.filtering_agent.check_query(query)
        if filter_result.get("obnoxious") or filter_result.get("prompt_injection"):
            return "Sorry, I cannot answer this question."
        if not filter_result.get("relevant"):
            return "Sorry, this is an irrelevant topic."

        # Query documents
        docs = self.query_agent.query_vector_store(query)

        # Generate response
        return self.answering_agent.generate_response(
            query,
            docs,
            conv_history,
            mode
        )

# ====================== Streamlit App ======================
if __name__ == "__main__":

    st.title("Mini Project 2: Streamlit Chatbot")
    mode = st.selectbox("Chat Mode", ["concise", "chatty", "funny"], key="mode_selector")

    open_ai_key_file = "../open_ai_key.txt" # Your OPEN AI Key in this file
    with open(open_ai_key_file, "r") as f:
      for line in f:
        openai_key = line
        break
    pinecone_key="pcsk_5STSET_PqapoTtnfMxmvpQS76fx8ZArgtqrgn6NyJVYaBZ4mUew6EqWALt87m2aCLwHFqE"
    pinecone_index_name = "similarity-search"

    # TODO: Define a function to get the conversation history
    def get_conversation() -> str:
        # return: A formatted string representation of the conversation.
        # ... (code for getting conversation history)
        return ""

    if "openai_model" not in st.session_state:
        # ... (initialize model)
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        # ... (initialize messages)
        st.session_state["messages"] = []

    # Display existing chat messages
    # ... (code for displaying messages)
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if "head_agent" not in st.session_state:
        st.session_state["head_agent"] = (
            Head_Agent(openai_key,
                       pinecone_key,
                       pinecone_index_name,
                       st.session_state["openai_model"]))

    # Wait for user input
    if prompt := st.chat_input("Ask me anything about machine learning!"):
        # ... (append user message to messages)
        st.session_state["messages"].append({"role": "user", "content": prompt})

        # ... (display user message)
        with st.chat_message("user"):
          st.write(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            # ... (send request to OpenAI API)
            # ... (get AI response and display it)
            with st.status("Analyzing query..."):
                response = st.session_state["head_agent"].main_loop(
                    prompt,
                    mode,
                    get_conversation(),
                )
            st.write(response)

        # ... (append AI response to messages)
        st.session_state["messages"].append({"role": "assistant", "content": response})