from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_fireworks import ChatFireworks
from tempfile import NamedTemporaryFile
from newspaper import Article

# Initialize the Fireworks LLM
llm = ChatFireworks(api_key="0YdGG4CL6KUAgR5v207G4kBb2rWvNkXoLDbyrHxc89ag3PVt", 
                    model="accounts/fireworks/models/llama-v3-8b-instruct")

# Function to load, split, and embed a document, and return a retriever using FAISS
def save_and_process_document(uploaded_file_path):
    loader = TextLoader(uploaded_file_path, encoding='utf-8')
    docs = loader.load()

    embeddings_model = CohereEmbeddings(cohere_api_key="uml0lVi8lxTjTL10Bkb42inOlNFk3zDf7sELxPDN",
                                        model="embed-english-light-v3.0")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1100, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    db = FAISS.from_documents(splits, embeddings_model)
    retriever = db.as_retriever(kwargs={"score_threshold": 0.5})
    return retriever

# Function to format a list of documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to fetch content from a website using newspaper3k
def fetch_website_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        content = article.text

        if content:
            return content, "Content fetched successfully using newspaper3k."
        else:
            return None, "Failed to fetch content using newspaper3k."
    except Exception as e:
        return None, f"Error fetching content: {str(e)}"

# Function to fetch content using Selenium when newspaper3k fails
def fetch_content_with_selenium(url):
    try:
        # Set up Selenium
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in headless mode
        service = Service('D:/Anvay/DL Submission/chromedriver-win64/chromedriver.exe')  # Update with your path
        driver = webdriver.Chrome(service=service, options=options)

        # Fetch the page
        driver.get(url)
        
        # Wait until the specific element that contains the relevant content is present
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "article")))  # Wait for the article tag to load
        
        # Get the rendered page source
        page_source = driver.page_source
        driver.quit()

        # Parse the content using BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')

        # Extract the text content from the specific section (e.g., within an article tag)
        content = soup.find('article').get_text(separator="\n")

        return content, "Content fetched successfully using Selenium."
    except Exception as e:
        return None, f"Error fetching content with Selenium: {str(e)}"

# Function to fetch hyperlinks from any website using Selenium
def fetch_hyperlinks(url):
    try:
        # Set up Selenium
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')  # Run in headless mode
        service = Service('D:/Anvay/DL Submission/chromedriver-win64/chromedriver.exe')  # Update with your path
        driver = webdriver.Chrome(service=service, options=options)

        # Fetch the page
        driver.get(url)
        
        # Wait for the page to load
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # Get the rendered page source
        page_source = driver.page_source
        driver.quit()

        # Parse the content using BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')

        # Extract all hyperlinks
        links = []
        for a_tag in soup.find_all('a', href=True):
            links.append(a_tag['href'])

        return links, "Hyperlinks fetched successfully."
    except Exception as e:
        return None, f"Error fetching hyperlinks: {str(e)}"

# Function to save text content to a file and process it
def save_content_to_file(content):
    # Save the fetched content to a temporary file
    with NamedTemporaryFile(delete=False, suffix=".txt", mode='w', encoding='utf-8') as temp_file:
        temp_file.write(content)
        temp_file_path = temp_file.name

    # Process the saved file and create a retriever
    return save_and_process_document(temp_file_path)

# Function to query the retriever and use the Fireworks model to generate answers
def llama_qna(llm, retriever, user_query):
    # Retrieve relevant documents from FAISS
    results = retriever.get_relevant_documents(user_query)
    if not results:
        print("No relevant documents found for the query.")
        return None

    # Format the relevant documents into a single string to use as context
    context = format_docs(results)

    # Construct the input prompt by combining the context and the query
    full_prompt = f"Context: {context}\n\nQuestion: {user_query}\nAnswer:"

    # Use the Fireworks model to generate an answer based on the context
    response = llm.predict(full_prompt)

    # Print and return the answer
    print("Answer from Fireworks Model:\n", response)
    return response
