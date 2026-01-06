# Build with docker build -t latentmas .
# Run with: docker run --gpus all -p 8000:8000 latentmas
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
RUN apt-get update && apt-get install -y python3-pip
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
# RUN pip3 install flash-attn --no-build-isolation # Build with docker build -t latentmas_hf .
COPY . .
EXPOSE 8000
CMD ["python3", "server.py", "--model_name", "Qwen/Qwen3-8B", "--devices", "cuda:0,cuda:1"]
# curl -X POST http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{"model": "latent-mas", "messages": [{"role": "user", "content": "Hello!"}], "mode": "normal", "add_think_token": true}'