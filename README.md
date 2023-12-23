---
title: Publico Demo
emoji: 🚀
colorFrom: green
colorTo: gray
sdk: gradio
python_version: 3.11.2
sdk_version: 4.12.0
app_file: app.py
pinned: true
fullWidth: true
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Publico.ai Demo

🚀 Check out the live demo at [Hugging Face Spaces](https://huggingface.co/spaces/PublicoDemo/Demo)

This project implements an interactive conversational chatbot to guide users through answering grant application questions.

## Description

The chatbot leads users through key stages like:

- Uploading documents
- Entering questions
- Checking comprehensiveness
- Generating final answers

It utilizes a `WorkflowManager` to handle state and progression between steps.

Responses are generated using LLMs like GPT-3.5-Turbo and GPT-4 via LangChain. Answers are constructed by ingesting and searching through the user's uploaded documents.

The system aims to create high quality grant application answers that cover all required aspects.

## Architecture

The app is structured into several key architectural groups:

### Workflow Management

- `chatbot_step.py`
- `chatbot_workflow.py`
- `step_decider.py`

These modules handle the definition of workflow steps, managing state and transitions between steps, and deciding the next step.

### Chatbot UI

- `app.py`
- `component_logic.py`
- `component_wrapper.py`

Launch the Gradio UI, handle component events and logic, and wrap components for extensibility.

### Context Tracking

- `context.py`

Tracks workflow state like user documents, questions, and answers.

### Response Generation

- `message_generator_*.py`
- `prompts.py`

Functions that generate chatbot responses via different approaches like LLMs. Also contains prompt engineering.

### LLM Integration

- `llm_streaming_utils.py`
- `openai_functions_utils.py`

Utilities for streaming text generation from LLMs and using advanced capabilities like OpenAI Functions.

### Document Processing

- `helpers.py`

Helper methods for ingesting, splitting, embedding, and searching user documents to construct answers.


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

docker run -it -p 7860:7860 --platform linux/amd64 \
  -d \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e CREATE_LINK=true \
  registry.hf.space/publicodemo-demo:latest python app.py
```

6. View logs:

```bash
docker container ls
docker logs <container_id>
```

### Configuration

Other environment variables can be set:

```bash
-e SERVER_PORT=5050 \
-e DEV=true \
-e CREATE_LINK=false \
-e EXCLUDE_LOGO=true \
-e CHATBOT_HEIGHT=800
-e CHATBOT_LAYOUT="bubble"
-e GPT_MODEL="gpt-3.5"
```
