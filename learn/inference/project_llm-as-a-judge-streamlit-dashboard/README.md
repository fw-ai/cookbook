## Project: Fireworks Model Comparison App

### Overview
The **Fireworks Model Comparison App** is an interactive tool built using **Streamlit** that allows users to compare various Large Language Models (LLMs) hosted on **Fireworks AI**. Users can adjust key model parameters, provide custom prompts, and generate model outputs to compare their behavior and responses. Additionally, an LLM-as-a-Judge feature is available to evaluate the generated outputs and provide feedback on their quality.

### Objectives
- **Compare Models**: Select different models from the Fireworks platform and compare their outputs based on a shared prompt.
- **Modify Parameters**: Fine-tune parameters such as **Max Tokens**, **Temperature**, **Top-p**, and **Top-k** to observe how they influence model behavior.
- **Evaluate Using LLM-as-a-Judge**: After generating responses, use a separate model to act as a judge and evaluate the outputs from the selected models.
  
### Features
- **Streamlit UI**: A simple and intuitive interface where users can select models, input prompts, and adjust model parameters.
- **LLM Comparison**: Select up to three different models, run a query with the same prompt, and view side-by-side responses.
- **Parameter Exploration**: Explore and modify different parameters such as Max Tokens, Temperature, Top-p, and more to see how they affect the model's response.
- **LLM-as-a-Judge**: Let another LLM compare the generated responses from the models and provide a comparison.

### App Structure
The app consists of two main pages:
1. **Comparing LLMs**:
   - Compare the outputs of three selected LLMs from Fireworks AI by providing a prompt.
   - View the responses side-by-side for easy comparison.
   - A selected LLM acts as a judge to evaluate the generated responses.
   
2. **Parameter Exploration**:
   - Modify various parameters for the LLMs (e.g., Max Tokens, Temperature, Top-p) and observe how they affect the outputs.
   - Compare three different outputs generated with varying parameter configurations.
   - Use LLM-as-a-Judge to provide a final evaluation of the outputs.

### Setup and Installation

#### Prerequisites
- **Python 3.x** installed on your machine.
- A **Fireworks AI** API key, which you can obtain by signing up at [Fireworks AI](https://fireworks.ai).
- Install **Streamlit** and the **Fireworks Python Client**.

#### Step-by-Step Setup
1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Set up a virtual environment (optional but recommended)**:
    ```bash
    python -m venv env
    source env/bin/activate  # On macOS/Linux
    .\env\Scripts\activate  # On Windows
    ```

3. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your environment variables**:
   - **Copy the `.env.template` file** into a new directory named `env`:
     ```bash
     mkdir env
     cp .env.template env/.env
     ```
   - **Rename the file** to `.env`:
     ```bash
     mv env/.env.template env/.env  # On macOS/Linux
     ren env\.env.template .env  # On Windows
     ```
   - **Fill in your Fireworks API key** in the `.env` file:
     ```bash
     FIREWORKS_API_KEY=<your_api_key>
     ```

5. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

6. **Explore the app**:
    - Open the app in your browser via the URL provided by Streamlit (typically `http://localhost:8501`).
    - Navigate between the pages to compare models and adjust parameters.

### Example Prompts
Here are some example prompts you can try in the app:
- **Prompt 1**: "Describe the future of AI in 500 words."
- **Prompt 2**: "Write a short story about a time traveler who visits ancient Rome."
- **Prompt 3**: "Explain quantum computing in simple terms."
- **Prompt 4**: "Generate a recipe for a healthy vegan dinner."

### Fireworks API Documentation
To learn more about how to query models and interact with the Fireworks API, visit the [Fireworks API Documentation](https://docs.fireworks.ai/api-reference/post-chatcompletions).

### Contributing
We welcome contributions to improve this app! To contribute, fork the repository, make your changes, and submit a pull request.

### License
This project is licensed under the MIT License.