import json
from image_downloader.downloader import download_images
from absl import app, flags


flags.DEFINE_string(
    "config",
    "image_downloader/config.json",
    "Path to Google Image Download config.json file",
)

FLAGS = flags.FLAGS


def main(_argv):
    with open(FLAGS.config, "r") as config:
        arguments = json.load(config)

    download_images(
        arguments["keywords"],
        arguments["chromedriver"],
        arguments["size"],
        arguments["limit"],
    )


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
