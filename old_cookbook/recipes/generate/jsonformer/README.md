# Constrained generation with Jsonformer

This recipe illustrates how to use [Jsonformer](https://github.com/1rgs/jsonformer) to constrain
the model output to a json string.

## Running instructions

We recommend running inside the preconfigured [Docker](https://github.com/fw-ai/cookbook/blob/main/recipes/docker/text/README.md) container.

```bash
pip install jsonformer
export PYTHONPATH=.:/workspace/cookbook # update to point to the cookbook repo root
python generate.py --config-name=api
```
