---
title: Publico Demo
emoji: 🚀
colorFrom: green
colorTo: gray
sdk: gradio
python_version: 3.11.2
sdk_version: 4.16.0
app_file: app.py
pinned: true
fullWidth: true
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Publico.ai Demo

🚀 Check out the live demo at [Hugging Face Spaces](https://huggingface.co/spaces/PublicoDemo/Demo)


## Description

Publico.ai is an interactive conversational chatbot designed to assist users in crafting comprehensive and compelling grant application answers.

The chatbot employs a `WorkflowManager` to handle state transitions and orchestrate the user journey through key stages, such as document uploading, question input, comprehensiveness evaluation, implicit question handling, user guidance, and final answer generation.

The system utilizes advanced Language Models via LangChain to generate responses. By ingesting and searching through the user's uploaded documents, it constructs high-quality grant application answers that cover all required aspects, ultimately helping users create compelling grant applications.


## Key Components and Files

### Workflow and Context Management (workflow)

Handles the definition of workflow steps, manages state and transitions between steps, decides the next step in the workflow, and tracks session state like user documents, questions, and answers.

- `chatbot_step.py`: Handles the definition of workflow steps.
- `chatbot_workflow.py`: Manages state and transitions between steps.
- `step_decider.py`: Decides the next step in the workflow.
- `session_state.py`: Tracks session state like user documents, questions, and answers.

### Chatbot UI (app.py, components)

Responsible for launching the Gradio UI and managing component events and logic.

- `app.py`: Main application file that launches the Gradio UI.
- `component_logic.py`: Handles component events and logic.
- `component_wrapper.py`: Wraps components for extensibility.

### Response Generation (message_generation, configurations)

Generates chatbot responses using different approaches like LLMs and contains prompt engineering.

- `msg_gen_publico.py`: Functions that generate chatbot responses using the Publico approach.
- `msg_gen_llm.py`: Functions that generate chatbot responses using LLMs (Language Models) like GPT-3.5-Turbo and GPT-4.
- `prompts.py`: Contains prompt engineering for LLMs.

### Text Processing and LLM Integration (utilities)

Provides utilities for streaming text generation from LLMs, using advanced capabilities like OpenAI Functions, and processing user documents to construct answers.

- `llm_streaming_utils.py`: Utilities for streaming text generation from LLMs.
- `openai_functions_utils.py`: Utilities for using advanced capabilities like OpenAI Functions.
- `document_helpers.py`: Helper methods for ingesting, splitting, embedding, and searching user documents to construct answers.


## Usage

### Local

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

### Docker

1. Download the private key in this repo `01-publico-ai-demo--key-pair.pem` and change its access permissions:

```bash
cd <directory-where-pem-file-is>
chmod 400 01-publico-ai-demo--key-pair.pem
```

2. SSH into the EC2 instance:

```bash
ssh -i "01-publico-ai-demo--key-pair.pem" ubuntu@ec2-54-219-71-193.us-west-1.compute.amazonaws.com
```

3. Change to superuser:
```bash
sudo su
```

4. Login to Docker registry:

```bash
docker login registry.hf.space
```

5. Pull latest image and run:

```bash
docker pull registry.hf.space/publicodemo-demo:latest

docker run --rm -it -p 7860:7860 --platform linux/amd64 \
  -d \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e SERVER_PORT=7860 \
  -e EXCLUDE_LOGO=true \
  -e CHATBOT_HEIGHT=560 \
  registry.hf.space/publicodemo-demo:latest python app.py
```

6. View logs:

```bash
docker container ls
docker logs <container_id>
```

### Configuration

Environment variables can be set to customize the app's behavior, such as server port, development mode, chatbot height, and layout, and GPT model selection.

```bash
-e SERVER_PORT=5050 \
-e DEV=true \
-e CREATE_LINK=false \
-e EXCLUDE_LOGO=true \
-e CHATBOT_HEIGHT=560
-e CHATBOT_LAYOUT="bubble"
-e GPT_MODEL="gpt-3.5"
```
