FROM python:3.10-bookworm
LABEL authors="Vladimir"

WORKDIR /streamlit_InstanceSegm

COPY . .
COPY models .
COPY predict .

RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "st_pred.py", "--server.port=8501", "--server.address=0.0.0.0"]