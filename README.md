# Fake Tweet Detection System
Python Flask app in Docker which detects whether a tweet is factually correct or inaccurate.

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
