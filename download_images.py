"""
while downloading images from google search scroll down to page manually for amount of
images to download
"""
import json
from absl import app, flags
from absl.flags import FLAGS
from data_generator.downloader import download_images
from data_generator.extract_frames import get_frames

flags.DEFINE_boolean("download", False, "Download image from google search")
flags.DEFINE_string(
    "config",
    "image_downloader/download_config.json",
    "Path to image download config.json file",
)
flags.DEFINE_string("output", "data", "path to output_frames")


def main(_argv):
    try:
        with open(FLAGS.config, "r") as config:
            arguments = json.load(config)
    except Exception as e:
        print("ERROR:: ", e)
    if FLAGS.download:
        download_images(
            arguments["keywords"],
            arguments["chromedriver"],
            FLAGS.output,
            arguments["size"],
            arguments["limit"],
            search_engine="duckduckgo",
        )

    else:
        get_frames(arguments["video_paths"], arguments["size"], FLAGS.output)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
