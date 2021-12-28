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
duckduckgo_base_url = "https://duckduckgo.com/?"


def load_image_from_url(url):
    img_request = requests.get(url)
    return Image.open(BytesIO(img_request.content))


def load_image_from_base64(base64_url):
    im_bytes = base64.b64decode(base64_url)
    return Image.open(BytesIO(im_bytes))


def google_download(driver, key, img_size, limit, saved_base_path):
    error = 0
    url_params = {"q": key, "tbm": "isch"}
    url = google_base_url + urllib.parse.urlencode(url_params)
    driver.get(url)
    WebDriverWait(driver, 100).until(
        EC.presence_of_element_located(
            (By.XPATH, '//*[@id="islmp"]/div/div/div/div/div[4]')
        )
    )
    time.sleep(1)
    try:
        box = driver.find_element_by_xpath('//*[@id="islrg"]/div[1]')
        frame = driver.find_element_by_id("islsp")
        all_images = box.find_elements_by_tag_name("img")[:limit]
        for i, img_tag in enumerate(all_images):
            img_src = img_tag.get_attribute("src")
            if img_src is not None:
                img = None
                if img_src.startswith("http"):
                    img = load_image_from_url(img_src)
                elif img_src.startswith("data:image/jpeg;base64,"):
                    img_tag.click()
                    img_link = frame.find_element_by_xpath(
                        '//*[@id="Sva75c"]/div/div/div[3]/div[2]/c-wiz/div[1]/div[1]/div/div[2]/a/img'
                    ).get_attribute("src")
                    if img_link:
                        img = load_image_from_url(img_link)
                    else:
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
        return len(all_images) - error
    except Exception as e:
        print(f"Cannot Load Page Correctly {e}")
        return None


def duckduckgo_download(driver, key, saved_base_path):
    error = 0
    url_params = {"q": key, "ia": "images", "iax": "images"}
    url = duckduckgo_base_url + urllib.parse.urlencode(url_params)
    driver.get(url)
    time.sleep(1)
    try:
        box = driver.find_element_by_xpath('//*[@id="zci-images"]/div[1]/div[2]')
        all_images = box.find_elements_by_tag_name("img")
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
                save_path = os.path.join(
                    saved_base_path, f"{key.replace(' ','_')}{i:003}.jpg"
                )
                img.save(save_path)
                print(f"{save_path} saved ")
            else:
                error += 1
        return len(all_images) - error
    except Exception as e:
        print(f"Cannot Load Page Correctly {e}")
        return None


def download_images(
    keywords,
    chromedriver,
    output_path,
    img_size=(1280, 720),
    limit=50,
    search_engine="duckduckgo",
):
    driver = webdriver.Chrome(chromedriver)
    if not os.path.isdir(os.path.join("data", "downloads")):
        os.mkdir(os.path.join("data", "downloads"))
    for key in keywords:
        if not os.path.isdir(os.path.join(output_path, "downloads", key)):
            os.mkdir(os.path.join(output_path, "downloads", key))
        saved_base_path = os.path.join(output_path, "downloads", key)
        t0 = time.time()
        if search_engine == "google":
            processed = google_download(driver, key, img_size, limit, saved_base_path)
            print(f"Total Image Processed: {processed}")
        elif search_engine == "duckduckgo":
            processed = duckduckgo_download(driver, key, saved_base_path)
            print(f"Total Image Processed: {processed}")
        t1 = time.time()
        print(f"Time Taken: {t1-t0}sec")
        time.sleep(random.random() * 0.2 + 0.1)
