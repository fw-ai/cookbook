# Building with Fireworks

This section contains examples optimized for learning and exploration of AI techniques. The projects here demonstrate patterns for building different kinds of applications using Fireworks AI. All examples are created and maintained by the Fireworks Developer Advocacy team, and each project will have its own instructions for installation and setup.

Browse through the sections below to explore these learning projects and get hands-on with Fireworks AI.

---

### Inference Projects

These projects focus on demonstrating how to perform model inference using Fireworks AI, showcasing various patterns and approaches for inference tasks.

| Project | Tools/Libraries Used | Description | Contributor | Use Case         | Additional Links | Project Type | 
|---------|----------------------|-------------|-------------|------------------|------------------|--------------|
| Fireworks Model Comparison App  | Streamlit, Fireworks Python API      | An interactive app for comparing LLMs hosted on Fireworks, with parameter tuning and LLM-as-a-Judge functionality.                  | @MMBazel        | Model Comparison, LLM Evaluation | [Project Link](./inference/project_llm-as-a-judge-streamlit-dashboard/README.md) | Notebook |
| Transcription Chat              | Next.js, Vercel, Google Fonts        | A Next.js app for transcription chat, utilizing Next.js features like font optimization and deployment via Vercel.                  | @benjibc        | Chatbot, Real-time Transcription | [Project Link](./inference/project_transcription_chat)     | App |
| Structured Response with Llama 3.1 | Fireworks Python Client, Pydantic | Demonstrates the use of structured responses in Llama 3.1, including Grammar Mode (GBNF) and JSON Mode for generating validated, consistent outputs for tasks like sentiment analysis and health records. | @aravindputrevu | Structured Data Generation, JSON Responses, Schema Validation | [Project Link](./structured_response/structured_response_llama_3_1.ipynb) | Notebook |
| Llama 3.1 Synthetic Data Generation | Fireworks Python Client, Pydantic | Demonstrates the use of Llama 3.1 models for generating synthetic data, including API usage and creating structured outputs for tasks like geography quizzes. | @aravindputrevu | Inference, Synthetic Data Generation | [Project Link](./synthetic_data/Llama_3_1_Synthetic_Data.ipynb) | Notebook |

---

## Fine-Tuning Projects

These projects focus on fine-tuning models with Fireworks AI. Each project highlights different approaches and tools used to adjust and optimize models based on custom datasets.

| Project | Tools/Libraries Used | Description | Contributor | Use Case | Additional Links | Project Type |
|---------|----------------------|-------------|-------------|----------|------------------|--------------|
|         |                      |             |             |          |                  |              |

---

## Function-Calling Projects

Here, you'll find projects that demonstrate how to implement function-calling capabilities using Fireworks AI, with different workflows for managing API calls and integrations.

| Project | Tools/Libraries Used | Description | Contributor | Use Case          | Additional Links | Project Type | 
|---------|----------------------|-------------|-------------|-------------------|------------------|--------------|
| Functional Chat Demo App        | Node.js, Fireworks, AlphaVantage, Quickchart, SDXL | A demo chat app with function-calling capabilities, enabling multi-turn conversations where functions perform tasks based on user inputs, including stock quotes, chart generation, and image generation. | @pgarbacki @benjibc  | Chatbot, Function Calling, Stock Prices, Image Generation | [Project Link](./function-calling/project_functional_chat)  | App |
| Fireworks LangChain Tool Usage | Fireworks Python Client, LangChain, OpenAI API | Demonstrates the use of Fireworks' Function-Calling capabilities integrated with LangChain for tool usage. Includes examples of invoking external tools like a custom calculator and handling chat-based queries. | @aravindputrevu | Function-Calling, Tool Integration, Agent Routing | [Project Link](./function-calling/notebooks_langchain/fireworks_langchain_tool_usage.ipynb) | Notebook |
| Fireworks LangGraph Tool Usage | Fireworks Python Client, LangGraph, LangChain | Demonstrates the use of Fireworks' Function-Calling capabilities integrated with LangGraph for complex tool routing. Includes examples of invoking custom tools like weather information and handling both chit-chat and tool-based queries using a state graph. | @aravindputrevu | Function-Calling, Graph-based Tool Integration, Agent Routing | [Project Link](./function-calling/notebooks_langchain/fireworks_langgraph_tool_usage.ipynb) | Notebook |
| Fireworks Function-Calling QA with OpenAI | Fireworks Python Client, OpenAI API | Demonstrates the use of Fireworks' Function-Calling capabilities for structured question-answering. Integrates OpenAI's API for enhanced responses and showcases multi-turn conversations with tool usage for precise data extraction and response formatting. | @aravindputrevu | Function-Calling, Structured Q&A, Multi-turn Conversations | [Project Link](./function-calling/notebooks_firefunction_openai/fireworks_firefunction_openai_qa.ipynb) | Notebook |
| Fireworks Function-Calling Demo | Fireworks Python Client, OpenAI API | Demonstrates a complete example of using Fireworks' Function-Calling API, including defining user queries, tool setup, and handling tool invocation. The notebook features a case study querying Nike's financial data for 2022 using a custom tool integration. | @aravindputrevu | Function-Calling, Financial Data Query, Tool Integration | [Project Link](./function-calling/notebooks_firefunction_openai/fireworks_function_calling_demo.ipynb) | Notebook |
| Fireworks Function-Calling for Information Extraction | Fireworks Python Client, OpenAI API, BeautifulSoup | Demonstrates using Fireworks' Function-Calling API for extracting structured information from web pages. Includes examples of extracting details about animals (e.g., Capybara, African Elephant) and summarizing news articles. | @aravindputrevu | Function-Calling, Information Extraction, Web Scraping | [Project Link](./function-calling/notebooks_firefunction_openai/fireworks_functions_information_extraction.ipynb) | Notebook |
| Fireworks AutoGen Stock Chart Demo | Fireworks Python Client, AutoGen, yFinance, Matplotlib | Demonstrates using Fireworks' Function-Calling API with the AutoGen framework to create an agent capable of generating stock price charts. Includes integration with yFinance for stock data retrieval and Matplotlib for chart visualization. | @aravindputrevu | Function-Calling, Financial Data Visualization, AutoGen Integration | [Project Link](./function-calling/notebooks_firefunction_openai/fw_autogen_stock_chart.ipynb) | Notebook |

---

## Retrieval-Augmented Generation (RAG) Projects

These examples showcase how to build RAG systems using Fireworks, focusing on the integration with external databases for enhanced document retrieval and response generation.

| Project | Tools/Libraries Used | Description | Contributor | Use Case         | Additional Links | Project Type |
|---------|----------------------|-------------|-------------|------------------|------------------|--------------|
| RAG Paper Title Generator      | Fireworks Python Client, ChromaDB, Sentence Transformers, Mistral-7B | An agentic system that leverages RAG and function-calling to generate short and catchy titles for research papers using embeddings and LLM-based completions. | @omarsar    | Research Title Simplification | [Project Link](./rag/project_simple_rag_chroma_paper_titles/rag-paper-titles.ipynb) | Notebook     |
| Project RAG with SurrealDB      | SurrealDB, Fireworks, FastAPI, Astro, TailwindCSS | A RAG app using SurrealDB for vector storage and Fireworks for LLM inference, enabling real-time knowledge updates and custom responses. | @aravindputrevu               | RAG, Dynamic Knowledge Updates | [Project Link](./rag/project_rag_with_surrealdb/README.md)            | App |
| Simple RAG with Chroma - League of Legends          | Chroma, Fireworks Python Client      | A simple Retrieval-Augmented Generation (RAG) system using Chroma as the vector store and Fireworks LLMs for embedding generation and context-aware responses. | @MMBazel     | RAG, Document Retrieval, Embedding Generation | [Project Link](./rag/project_simple-rag-with-chroma/README.md)         | Notebook |
| MongoDB RAG Movie Recommender  | Fireworks Python Client, MongoDB Atlas, OpenAI API | An agentic movie recommendation system using Fireworks' function-calling models for personalized, real-time recommendations based on user queries and MongoDB vector search. | @aravindputrevu    | Personalized Recommendations, Real-time Query Handling | [Project Link](../integrations/MongoDB/project_rag_with_mongodb/README.md) | Notebook     |

## Agentic Systems Projects

These projects explore agentic systems built with Fireworks, leveraging function-calling models to create autonomous decision-making workflows.

| Project | Tools/Libraries Used | Description | Contributor | Use Case | Additional Links | Project Type |
|---------|----------------------|-------------|-------------|----------|------------------|--------------|
|         |                      |             |             |          |                  |              |

---

## Compound Systems Projects

In these projects, you’ll learn how to build compound systems using Fireworks, where different models and workflows interact to solve complex tasks in a unified architecture.

| Project | Tools/Libraries Used | Description | Contributor | Use Case | Additional Links | Project Type |
|---------|----------------------|-------------|-------------|----------|------------------|--------------|
|         |                      |             |             |          |                  |              |

---

## How to Submit Your Project

If you've built an example that fits into this section, we'd love to feature it in our repository! To submit your project, follow our [Contribution Guide](../Contribution.md). Ensure that each project includes:
- A `README.md` with setup and usage instructions.
- A clear description of how Fireworks is used in the project.