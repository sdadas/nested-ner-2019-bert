FROM nvcr.io/nvidia/pytorch:20.03-py3
RUN pip install -U setuptools pip && pip install numpy>=1.17.3 adabound>=0.0.5 stanfordnlp>=0.2.0 transformers>=2.8.0