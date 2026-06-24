# Use a clean, modern base. Ubuntu 22.04 is fine, but we optimize it.
FROM ubuntu:22.04

# Prevent interactive prompts blocking builds
ENV DEBIAN_FRONTEND=noninteractive
# Default to glx for GUI rendering via X11 forwarding
ENV MUJOCO_GL=glx
ENV PYTHONUNBUFFERED=1
ENV PATH="/home/robotics/.local/bin:${PATH}"

# Combine update, install, and cache cleanup in a single layer to minimize size
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglx-mesa0 \
    libosmesa6-dev \
    libglfw3 \
    libglew-dev \
    patchelf \
    ffmpeg \
    git \
    python3.10 \
    python3-pip \
    python3-venv \
    x11-apps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Dynamically set build arguments to match host user permissions
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create a dedicated non-root engineer user
RUN groupadd -g ${GROUP_ID} robotics && \
    useradd -m -u ${USER_ID} -g robotics -s /bin/bash robotics

USER robotics
WORKDIR /home/robotics/workspace

# Leverage Docker cache: install dependencies FIRST
COPY --chown=robotics:robotics requirements.txt .
RUN pip3 install --no-cache-dir --user -r requirements.txt

# Copy the actual codebase. Changes here will not trigger a reinstall of pip packages.
COPY --chown=robotics:robotics . .

# Drop the hardcoded entrypoint flags. Control execution via the runner script or compose.
ENTRYPOINT ["python3"]