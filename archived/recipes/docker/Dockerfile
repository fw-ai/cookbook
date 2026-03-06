ARG CUDA_VERSION="12.4.1"
ARG UBUNTU_VERSION="22.04"

FROM nvidia/cuda:$CUDA_VERSION-cudnn-devel-ubuntu$UBUNTU_VERSION

ENV PATH="/root/miniconda3/bin:${PATH}"

ARG PYTHON_VERSION="3.10"
ARG CUDA="124"

ENV PYTHON_VERSION=$PYTHON_VERSION

RUN apt-get update \
    && apt-get install -y \
    curl \
    git \
    build-essential \
    ninja-build \
    git-lfs \
    libaio-dev \
    libpq-dev \
    psmisc \
    && rm -rf /var/lib/apt/lists/* \
    && curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "Miniconda3-latest-Linux-x86_64.sh" \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda create -n "py${PYTHON_VERSION}" python="${PYTHON_VERSION}"

ENV PATH="/root/miniconda3/envs/py${PYTHON_VERSION}/bin:${PATH}"

WORKDIR /workspace

RUN curl "https://storage.googleapis.com/fireworks-public/firectl/stable/linux-amd64.gz" -o "firectl.gz" && \
    gunzip firectl.gz && \
    install -o root -g root -m 0755 firectl /usr/local/bin/firectl && \
    rm firectl

# install gcloud cli and opencv deps
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && \
    apt-get install -y google-cloud-cli libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6

RUN python3 -m pip install --upgrade pip && pip3 install packaging && \
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA}

RUN git lfs install --skip-repo

RUN pip install \
    accelerate \
    anthropic \
    astor \
    awscli \
    backoff \
    beautifulsoup4 \
    bitsandbytes \
    datamodel-code-generator \
    datasets \
    deepspeed \
    evals \
    evaluate \
    fire \
    fireworks-ai \
    flash_attn \
    jsonref \
    gcsfs \
    google-cloud \
    google-cloud-bigquery \
    google-cloud-logging \
    google-cloud-pubsub \
    google-cloud-storage \
    google-generativeai \
    guidance \
    huggingface_hub \
    hydra-core \
    jupyter \
    mammoth \
    markdownify \
    nbconvert \
    ninja \
    packaging \
    paddlepaddle \
    paddleocr \
    peft \
    pillow \
    plotly \
    # TODO(yifan): please recompile proto files with a newer protoc
    protobuf==3.20.* \
    psycopg2 \
    py7zr \
    PyMuPDF \
    PyPDF2 \
    pydantic==2.4.2 \
    pytablewriter \
    PyYAML \
    s3fs \
    sentencepiece \
    sentence_transformers \
    # workaround for the build failure
    spacy!=3.0.6 \
    sqlitedict \
    tabulate \
    torchx \
    transformers \
    trl \
    wandb \
    zstandard

RUN pip install \
    llama-index-core \
    llama-index-embeddings-huggingface \
    llama-index-finetuning \
    llama-index-llms-fireworks \
    llama-index-readers-file \
    pypdf

RUN pip install --no-cache-dir pdfservices-sdk

