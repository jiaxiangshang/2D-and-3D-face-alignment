FROM kaixhin/cuda-torch:8.0

# Tencent
# Setup environment.
# Default command on startup.
#CMD bash
#ENV http_proxy "http://web-proxy.tencent.com:8080"
#ENV https_proxy "http://web-proxy.tencent.com:8080"
#ENV http_proxy 'http://127.0.0.1:12639'
#ENV https_proxy 'http://127.0.0.1:12639'
#RUN export http_proxy=http://127.0.0.1:12639
#RUN export https_proxy=http://127.0.0.1:12639
#RUN export no_proxy=mirrors.tencent.com,mirrors.cloud.tencent.com,localhost,127.0.0.1,.oa.com,.tencent-cloud.com,.tencentyun.com
#RUN echo 'Acquire::http::proxy "http://web-proxy.oa.com:8080";' >> /etc/apt/apt.conf
#RUN echo 'Acquire::https::proxy "http://web-proxy.oa.com:8080";' >> /etc/apt/apt.conf
#RUN echo 'Acquire::http::Proxy "http://127.0.0.1:12639";' >> /etc/apt/apt.conf
#RUN echo 'Acquire::https::Proxy "http://127.0.0.1:12639";' >> /etc/apt/apt.conf
#RUN echo 'Acquire::http::Proxy::mirrors.tencent.com DIRECT;' >> /etc/apt/apt.conf
#RUN echo 'Acquire::https::Proxy::mirrors.tencent.com DIRECT;' >> /etc/apt/apt.conf

# 10.11.56.22 10.11.56.23 10.28.0.12 10.14.0.130

#
#RUN echo 'deb http://mirrors.163.com/debian/ jessie main non-free contrib' > /etc/apt/sources.list
#RUN echo 'deb http://mirrors.163.com/debian/ jessie-updates main non-free contrib' >> /etc/apt/sources.list
#RUN echo 'deb http://mirrors.163.com/debian-security/ jessie/updates main non-free contrib' >> /etc/apt/sources.list
#RUN echo "deb http://mirrors.tencent.com/ubuntu/ bionic main restricted">>/etc/apt/sources.list
#RUN echo "deb http://mirrors.tencent.com/debian/ bionic main restricted">>/etc/apt/sources.list
#RUN echo "deb http://mirrors.tencent.com/debian/ bionic main restricted">>/etc/apt/sources.list

# Install depenencies and python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python-numpy \
    python-matplotlib \
    libmatio2 \
    libgoogle-glog-dev \
    libboost-all-dev \
    python-dev \
    python-tk

#RUN pip install dlib

# Install lua packages
RUN luarocks install xlua &&\
    luarocks install matio

# Build thpp
WORKDIR /opt
RUN git clone https://github.com/facebook/thpp
WORKDIR /opt/thpp
RUN git fetch origin pull/33/head:NEWBRANCH && git checkout NEWBRANCH
WORKDIR /opt/thpp/thpp
RUN THPP_NOFB=1 ./build.sh

# Build fb.python
WORKDIR /opt
RUN git clone https://github.com/facebook/fblualib
WORKDIR /opt/fblualib/fblualib/python
RUN luarocks make rockspec/*

# Clone our repo
WORKDIR /workspace
RUN chmod -R a+w /workspace
RUN git clone https://github.com/1adrianb/2D-and-3D-face-alignment

# nmcli dev show | grep 'IP4.DNS'
# docker run busybox nslookup www.baidu.com
# docker run --dns 192.168.0.1 busybox nslookup www.baidu.com
# 10.11.56.22 10.11.56.23 10.28.0.12 10.14.0.130
# DOCKER_OPTS="--dns 8.8.8.8 --dns 8.8.4.4 --dns 10.11.56.22 --dns 10.11.56.23 --dns 10.28.0.12 --dns 10.14.0.130"
#ENV http_proxy ""
#ENV https_proxy ""