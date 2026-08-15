# Dockerfile
FROM continuumio/miniconda3:24.9.2-0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    CONDA_ENV=nces-benchmark

# Set (force) the library path to include the conda environment's lib directory, which is necessary for some libraries to function correctly.
ENV LD_LIBRARY_PATH=/opt/conda/envs/nces-benchmark/lib:$LD_LIBRARY_PATH

# Java is required by owlapy's OWLAPI synchronisation layer (HermiT/Pellet).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        unzip \
        openjdk-17-jre-headless \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

WORKDIR /app

COPY environment.yml requirements.txt ./
RUN conda env create -f environment.yml && conda clean -afy

# Make the conda env the default interpreter for every later layer.
SHELL ["conda", "run", "--no-capture-output", "-n", "nces-benchmark", "/bin/bash", "-c"]
ENV PATH=/opt/conda/envs/nces-benchmark/bin:$PATH

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-deps -e .

COPY input/ ./input/
COPY tests/ ./tests/
COPY datasets/ ./datasets/

RUN mkdir -p /app/Output

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "nces-benchmark", "nces-benchmark"]
CMD ["--help"]
