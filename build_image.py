import click
import docker
from .utils import file_relative_path

IMAGE_NAME = "gyozasimulator"


@click.command()
@click.option("--image_name", default=IMAGE_NAME, help="tag for the resulting docker image")
@click.option("--tag_name", help="tag for the resulting docker image")
def build_image(image_name: str, tag_name: str):
    """Builds the docker image for the Serverless Compute Simulator"""
    client = docker.from_env()
    dockerfile_directory = file_relative_path(__file__, "./")
    client.images.build(path=dockerfile_directory, tag=f"{image_name}:{tag_name}")


if __name__ == "__main__":
    build_image()
