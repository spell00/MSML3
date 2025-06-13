FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Toronto

# Install system dependencies including R
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libssl-dev \
        r-base \
        r-base-dev \
        r-cran-devtools \
        tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# Copy all necessary files
COPY msml/ /app/msml/
COPY msml/tests/ /app/msml/tests/
COPY requirements.txt requirements-dev.txt pytest.ini /app/
COPY setup.py /app/

# Install both production and development dependencies
RUN pip install -r /app/requirements.txt && \
    pip install -r /app/requirements-dev.txt && \
    pip install -e .

ENV R_HOME=/usr/lib/R
ENV LD_LIBRARY_PATH=/usr/lib/R/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/app:${PYTHONPATH}

# The pytest command is now configured in pytest.ini
CMD ["pytest"]
