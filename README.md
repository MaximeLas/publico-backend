---
title: Publico Demo
emoji: 🚀
colorFrom: green
colorTo: gray
sdk: gradio
python_version: 3.11.2
sdk_version: 3.42.0
app_file: app.py
pinned: true
fullWidth: true
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Publico.ai Demo

## To run locally

```
pip install -r requirements.txt
python app.py
```

## To launch in AWS EC2

1. Download the private key in this repo - `01-publico-ai-demo--key-pair.pem`

```
chmod 400 01-publico-ai-demo--key-pair.pem
```

2. Open terminal session locally:

```
cd <your-directory-where-pem-file-is>

ssh -i "01-publico-ai-demo--key-pair.pem" ec2-user@ec2-54-151-9-20.us-west-1.compute.amazonaws.com
```

3. Once logged in change to superuser:

```
sudo su
```

4. Make sure docker is installed. For that, you can follow [these docs](https://docs.docker.com/engine/install/ubuntu/).

5. Log in to registry.hf.space, pull the latest publicodemo-demo image, and run the app!

```
# login is your username
# password has to be an Access Token you create for your account
# https://huggingface.co/docs/hub/spaces-run-with-docker
docker login registry.hf.space

docker image pull registry.hf.space/publicodemo-demo:latest

docker run -it -p 7860:7860 --platform=linux/amd64 \
	-e OPENAI_API_KEY="YOUR_VALUE_HERE" \
	-e CREATE_LINK=true
	registry.hf.space/publicodemo-demo:latest python app.py
```

## To view container logs

1. SSH into EC2.

2. Check which container is running the image you want by listing all the logs.

```
docker container ls
```

3. Output the logs

```
docker logs d54dfdf72f1d
```
