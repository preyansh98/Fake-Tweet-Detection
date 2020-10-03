FROM python:3.6
LABEL maintainer="preyansh98@gmail.com"
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
RUN [ "python", "-c", "import nltk; nltk.download('all')" ]
ENTRYPOINT ["python"]
CMD ["app/app.py"]
