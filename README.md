# Fake Tweet Detection System
Python Flask app in Docker which detects whether a tweet is factually correct or inaccurate. This app is an interactive dashboard that displays the flow of truths and lies that stream through the platform each day, using NLP to flag tweets that may contain misinformation.

POST: http://34.123.199.55/predict_tweet with a json {"tweet": "the tweet text goes here"} and will receive a json {"result":"fake"}

### Build application
Build the Docker image manually by cloning the Git repo.
```
$ sudo docker build -t gcr.io/fake-tweet-detection/fake-tweet-backend:v1 ./
```


### Run the container
Create a container from the image.
```
$ sudo docker run --rm -p 8080:8080 gcr.io/fake-tweet-detection/fake-tweet-backend:v1
```

### Push the container to cloud
Push the container to GCP:
```
$ sudo docker push gcr.io/fake-tweet-detection/fake-tweet-backend:v1
```

Now visit http://localhost:8080
