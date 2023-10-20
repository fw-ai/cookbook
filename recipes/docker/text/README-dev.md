# Docker container dev book

## Publish container to Docker Hub

```bash
docker login fwai
docker build -t fwai/cookbook:latest -t fwai/cookbook:$(cat ../../../VERSION) .
fwai/cookbook:$(cat ../../../VERSION)
docker push fwai/cookbook:$(cat ../../../VERSION)
docker push fwai/cookbook:latest
```
