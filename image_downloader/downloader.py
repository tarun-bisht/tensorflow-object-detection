from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from io import BytesIO
from PIL import Image
import urllib.parse
import os
import time
import base64
import random
import requests

google_base_url = "https://www.google.com/search?"


def load_image_from_url(url):
    img_request = requests.get(url)
    return Image.open(BytesIO(img_request.content))


def load_image_from_base64(base64_url):
    im_bytes = base64.b64decode(base64_url)
    return Image.open(BytesIO(im_bytes))


def complete_loading(driver):
    try:
        return 0 == driver.execute_script("return jQuery.active")
    except Exception as e:
        print("ERROR: ", e)


"""https://stackoverflow.com/questions/20986631/how-can-i-scroll-a-web-page-using-selenium-webdriver-in-python"""


def scroll_bottom(driver):
    SCROLL_PAUSE_TIME = 0.1
    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        # Wait to load page
        time.sleep(SCROLL_PAUSE_TIME)
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def download_images(keywords, chromedriver, img_size=(1280, 720), limit=50):
    driver = webdriver.Chrome(chromedriver)
    if not os.path.isdir("download"):
        os.mkdir("download")
    for key in keywords:
        if not os.path.isdir(os.path.join("download", key)):
            os.mkdir(os.path.join("download", key))
        saved_base_path = os.path.join("download", key)
        t0 = time.time()
        error = 0
        google_url_params = {"q": key, "tbm": "isch"}
        url = google_base_url + urllib.parse.urlencode(google_url_params)
        driver.get(url)
        scroll_bottom(driver)
        WebDriverWait(driver, 100).until(
            EC.presence_of_element_located(
                (By.XPATH, '//*[@id="islmp"]/div/div/div/div/div[4]')
            )
        )
        time.sleep(1)
        try:
            box = driver.find_element_by_id("islmp")
            all_images = box.find_elements_by_tag_name("img")[:limit]
            for i, img in enumerate(all_images):
                img_src = img.get_attribute("src")
                if img_src is not None:
                    img = None
                    if img_src.startswith("http"):
                        img = load_image_from_url(img_src)
                    elif img_src.startswith("data:image/jpeg;base64,"):
                        img = load_image_from_base64(
                            img_src.replace("data:image/jpeg;base64,", "")
                        )
                    img = img.convert(mode="RGB")
                    img.thumbnail(img_size)
                    save_path = os.path.join(saved_base_path, f"{key}{i:003}.jpg")
                    img.save(save_path)
                    print(f"{save_path} saved ")
                else:
                    error += 1
            t1 = time.time()
            print(f"Time Taken: {t1-t0}sec")
            print(f"Total Image Processed: {len(all_images)}")
            print(f"URL with Error: {error}")
            print(f"Total Images Saved: {len(all_images)-error}")
            time.sleep(random.random() * 0.2 + 0.1)
        except Exception as e:
            print(f"Cannot Load Page Correctly {e}")
