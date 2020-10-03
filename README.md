# python-flask-docker
Basic Python Flask app in Docker which prints the hostname and IP of the container

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

Now visit http://localhost:8080
```
 The hostname of the container is 6095273a4e9b and its IP is 172.17.0.2. 
```

