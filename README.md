---
title: Publico Demo
emoji: 🚀
colorFrom: green
colorTo: gray
sdk: gradio
python_version: 3.11.2
sdk_version: 3.44.3
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

1. Download the private key in this repo – `01-publico-ai-demo--key-pair.pem` – and change its access permissions:

```
chmod 400 01-publico-ai-demo--key-pair.pem
```

2. Open terminal session locally:

```
cd <directory-where-pem-file-is>

ssh -i "01-publico-ai-demo--key-pair.pem" ubuntu@ec2-54-219-71-193.us-west-1.compute.amazonaws.com
```

3. Once logged in change to superuser:

```
sudo su
```

4. Make sure docker is installed. If it's not, you can follow [these docs](https://docs.docker.com/engine/install/ubuntu/):

```
for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do sudo apt-get remove $pkg; done && # Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add the repository to Apt sources:
echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update && sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && sudo docker run hello-world
```

5. Log in to registry.hf.space, pull the latest publicodemo-demo image, and run the app!

```
# login:    your username
# password: an Access Token you create for your account
# https://huggingface.co/docs/hub/spaces-run-with-docker
docker login registry.hf.space

docker run -it -p 7860:7860 --platform=linux/amd64 \
	-d \
	-e OPENAI_API_KEY="YOUR_KEY_HERE" \
	-e CREATE_LINK=true \
	registry.hf.space/publicodemo-demo:latest python app.py

# Can also specify a different port
docker run -it -p 5050:5050 --platform=linux/amd64 \
	-d \
	-e OPENAI_API_KEY="YOUR_KEY_HERE" \
	-e CREATE_LINK=true \
	-e SERVER_PORT=5050 \
	registry.hf.space/publicodemo-demo:latest python app.py
```

## To view container logs in AWS EC2

1. SSH into EC2.

2. Check which container is running the image you want by listing all the logs.

```
docker container ls
```

3. Output the logs

```
docker logs d54dfdf72f1d
```
