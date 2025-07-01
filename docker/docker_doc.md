## build
docker build --platform linux/arm64 -t inference:latest -f docker/Dockerfile .

## inspect
docker run -it --rm inference:latest /bin/bash
 used to inspect the image

## run
docker run -p 8000:8000 --name inference_container inference:latest

docker run	Start a new container based on an image
-p 8000:8000	Map port 8000 on your host → port 8000 in the container
--name inference_container	Give the container a custom, readable name: inference_container
inference:latest	Use the Docker image named inference, with the latest tag

docker run --network bridge -it --rm inference:latest /bin/sh
to give network acccess (might be dangerous)


How to use it:
Save that as docker-compose.yml in your project root (the same place your Dockerfile is).

Run:

bash
Copy
Edit
docker-compose up --build
This will:

Build your Docker image based on the Dockerfile in .

Start the container named inference_container

Map port 8000 of the container to port 8000 on your host

Access your API at http://localhost:8000

Additional tips:
If you want to run it detached (in background):

bash
Copy
Edit
docker-compose up -d --build
To stop the service:

bash
Copy
Edit
docker-compose down
