FROM nvcr.io/nvidia/pytorch:19.10-py3
RUN pip install -U setuptools pip && pip install numpy>=1.17.3 adabound>=0.0.5 stanfordnlp>=0.2.0 transformers>=2.1.1