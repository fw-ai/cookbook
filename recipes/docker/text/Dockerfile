FROM nvcr.io/nvidia/pytorch:23.09-py3 as efs-utils-build

WORKDIR /
RUN apt-get update && apt-get -y install git binutils
RUN git clone https://github.com/aws/efs-utils
WORKDIR /efs-utils
RUN ./build-deb.sh

FROM nvcr.io/nvidia/pytorch:23.09-py3

WORKDIR /

COPY --from=efs-utils-build /efs-utils/build/amazon-efs-utils*deb /tmp/amazon-efs-utils.deb

# keep alphabetical order
RUN apt-get update && apt-get install --no-install-recommends -y \
    ca-certificates \
    curl \
    git-lfs \
    libaio1 \
    libaio-dev \
    unzip \
    /tmp/amazon-efs-utils.deb

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm awscliv2.zip

RUN curl "https://storage.googleapis.com/fireworks-public/firectl/stable/linux-amd64.gz" -o "firectl.gz" && \
    gunzip firectl.gz && \
    install -o root -g root -m 0755 firectl /usr/local/bin/firectl && \
    rm firectl

RUN pip install --upgrade pip

# keep alphabetical order
RUN pip install \
    accelerate \
    backoff \
    bitsandbytes \
    datasets \
    evals \
    fire \
    fireworks-ai \
    guidance \
    huggingface_hub \
    hydra-core \
    jupyter \
    nbconvert \
    ninja \
    packaging \
    peft \
    py7zr \
    s3fs \
    sentencepiece \
    sentence_transformers \
    torchx \
    transformers \
    wandb \
    zstandard

RUN pip uninstall -y transformer-engine

RUN pip install --upgrade flash-attn --no-build-isolation

ARG recipes_commit=main
RUN git clone https://github.com/fw-ai/cookbook /workspace/cookbook && \
    cd /workspace/cookbook && \
    git checkout ${recipes_commit}
