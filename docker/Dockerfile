FROM osrf/ros:humble-desktop

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
       x11-apps \
       vim \
       net-tools \
       sudo \
       cmake \
       libgl1-mesa-dev \
       libxinerama-dev \
       libxcursor-dev \
       libxrandr-dev \
       libxi-dev \
       ninja-build \
       zlib1g-dev \
       clang-12 \
       binutils \
       build-essential \
       llvm-12 \
  && rm -rf /var/lib/apt/lists/*

ARG USER_ID=1046
ARG GROUP_ID=1046
RUN groupadd -g ${GROUP_ID} dimos \
    && useradd -u ${USER_ID} -g ${GROUP_ID} -s /bin/bash -m -d /home/dimos dimos \
    && echo 'dimos ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN ln -sf /usr/bin/clang-12 /usr/bin/clang \
  && ln -sf /usr/bin/clang++-12 /usr/bin/clang++ \
  && if [ -f /usr/bin/llvm-ar-12 ]; then ln -sf /usr/bin/llvm-ar-12 /usr/bin/llvm-ar; fi \
  && if [ -f /usr/bin/llvm-ranlib-12 ]; then ln -sf /usr/bin/llvm-ranlib-12 /usr/bin/llvm-ranlib; fi

ENV CMAKE_C_COMPILER_AR=/usr/bin/ar
ENV CMAKE_CXX_COMPILER_AR=/usr/bin/ar
ENV CMAKE_C_COMPILER_RANLIB=/usr/bin/ranlib
ENV CMAKE_CXX_COMPILER_RANLIB=/usr/bin/ranlib

# Create a helper script to configure cmake with correct archiver tools
RUN echo '#!/bin/bash' > /usr/local/bin/cmake-configure && \
    echo 'cmake -DCMAKE_C_COMPILER_AR=/usr/bin/ar \ ' >> /usr/local/bin/cmake-configure && \
    echo '      -DCMAKE_CXX_COMPILER_AR=/usr/bin/ar \ ' >> /usr/local/bin/cmake-configure && \
    echo '      -DCMAKE_C_COMPILER_RANLIB=/usr/bin/ranlib \ ' >> /usr/local/bin/cmake-configure && \
    echo '      -DCMAKE_CXX_COMPILER_RANLIB=/usr/bin/ranlib \ ' >> /usr/local/bin/cmake-configure && \
    echo '      "$@"' >> /usr/local/bin/cmake-configure && \
    chmod +x /usr/local/bin/cmake-configure

SHELL ["/bin/bash", "-lc"]

USER dimos
WORKDIR /home/dimos

CMD ["bash"]
