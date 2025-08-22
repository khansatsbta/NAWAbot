# NAWAbot
**Nawabot** is a Retrieval-Augmented Generation (RAG) program that responds to all user inquiries regarding Nawatech by allowing users to communicate with the bot via a chatbot interface.  The system uses sophisticated embedding models (Ollama) and a local vector storage for effective and precise question-answering. 

## LLM DeepSeek API Key Needed!!!

## Features
- RAG workflow that combines retrieval and generation for high-quality responses.
- Secure functions to remove common injection phrases and to limit how many times a user can make a request in a given time period.
- Splits text into manageable chunks.
- Creates embeddings and a vector store using Chroma.
- Interactive interface with Streamlit Interface

---

## Installation & Setup

Follow the steps below to set up and run the application:

### 1. Prerequisites
Ensure you have the following installed on your system:
- Docker: [Get Docker](https://docs.docker.com/get-started/get-docker/)
- Docker Compose: Usually included with Docker Desktop

### 2. Project Structure

```
/NAWAbot/
├── app.py
├── FAQ_Nawa.pdf
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

### 3. Set Your API Key

Input your DeepSeek API key in the app.py configuration.

```app.py
DEEPSEEK_API_KEY = "Put your API key here"
```

### 4. Launch the Application

Open a new terminal window.

```bash
docker-compose up --build
```

### 5. Pull the Ollama Model (One-Time Setup)

Open a new terminal window, keep the previous terminal.
```bash
docker exec -it ollama_service ollama pull nomic-embed-text
```
Only need to do this once. The model will be saved in a persistent volume.
### 6. Access the Chatbot

Once the containers are running and the model is downloaded, open your web browser and navigate to:
```
http://localhost:8501
```

### 7. Stop the Application
To stop the chatbot and shut down all containers, return to the terminal where you ran docker-compose up and press Ctrl + C. Then, run the following command to clean up:
```bash
docker-compose down
```

## Configuration

You can easily modify the following parameters at the top of the `app.py` file:

1. **Models**:
   - `EMBEDDING_MODEL`: Change the local model used for embeddings.
   - `DEEPSEEK_MODEL`: Change the chat model used for generation.

2. **Security**:
   - `COOLDOWN_PERIOD`: Adjust the rate-limiting cooldown time (in seconds).

3. **Data Resource**:
   - `FAQ_FILE_PATH`: Change the name of the Excel file to use as the knowledge base.


## Troubleshooting

1. **"API Key Not Set" Error**: Make sure you set the DEEPSEEK_API_KEY environment variable in your terminal before running docker-compose up.

2. **"Connection Refused" to Ollama**: Ensure the base_url in app.py is set to http://ollama:11434. This is the internal network address for the Ollama service.

3. **Build Fails**: If the build fails on pip install, double-check that your requirements.txt file is correct and that the build-essential package is being installed in your Dockerfile.

4. **Check Container Logs**: If the application is not behaving as expected, check the logs for errors:
```bash
docker-compose logs -f app
```
