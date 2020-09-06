FROM tensorflow/tensorflow:1.12.0-gpu-py3
ENV PATH /opt/conda/bin:$PATH
RUN apt-get update && apt-get install -y curl
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

RUN conda update -y conda
RUN apt-get update
RUN apt-get install -y python3-pip
RUN apt-get install -y vim
RUN conda install python==3.6.7
COPY ./config/conda_gpu.yml /usr/conda_gpu.yml
RUN conda env update --file /usr/conda_gpu.yml

ARG USER_ID
ARG GROUP_ID

WORKDIR /usr/toffs_train

RUN if [ ${USER_ID:-0} -ne 0 ] && [ ${GROUP_ID:-0} -ne 0 ]; then \
    groupadd -g ${GROUP_ID} toffs &&\
    useradd -l -u ${USER_ID} -g toffs toffs &&\
    chown --changes --silent --no-dereference --recursive \
          --from=1000:1000 ${USER_ID}:${GROUP_ID} \
        /usr/toffs_train \
;fi

USER toffs

ENTRYPOINT ["python","-m", "src.train"]
