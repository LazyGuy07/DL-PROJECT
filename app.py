import streamlit as st
from content_processing import fetch_website_content, fetch_content_with_selenium, save_content_to_file, llama_qna

def main():
    st.title("Q&A Chatbot with Web Content")

    # Input for the URL
    url = st.text_input("Enter a website URL:", key="url_input")

    if st.button("Fetch Content"):
        if url:
            # Attempt to fetch content using newspaper3k
            content, message = fetch_website_content(url)

            if content is None:
                # If newspaper3k fails, try fetching content with Selenium
                st.warning("Newspaper3k failed, trying Selenium...")
                content, message = fetch_content_with_selenium(url)

            st.success(message)

            if content:
                # Save content to file and create a retriever
                retriever = save_content_to_file(content)
                st.success("Content processed successfully. You can now ask questions.")

                # Input for user query
                query = st.text_input("Enter a query (or type 'exit' to quit):", key="query_input")

                if query and query.lower() != "exit":
                    # Get answer from the Fireworks model
                    answer = llama_qna(llm, retriever, query)
                    st.write("Answer:", answer)
            else:
                st.error("Failed to fetch content.")
        else:
            st.error("Please enter a valid URL.")

if __name__ == "__main__":
    main()

