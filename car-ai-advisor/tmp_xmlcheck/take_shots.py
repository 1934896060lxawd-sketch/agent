"""Edge headless 截图：欢迎页 + 完整问答页。"""
import time, tomllib
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
PWD = tomllib.loads((ROOT / ".streamlit" / "secrets.toml").read_text(encoding="utf-8")).get("ACCESS_PASSWORD", "")

opts = Options()
opts.binary_location = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
opts.add_argument("--headless=new")
opts.add_argument("--window-size=1280,960")
opts.add_argument("--disable-gpu")
opts.add_argument("--lang=zh-CN")
driver = webdriver.Edge(opts)
wait = WebDriverWait(driver, 30)

try:
    driver.get("http://localhost:8501")
    time.sleep(4)

    # 密码门
    pwd = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']")))
    pwd.send_keys(PWD)
    pwd.send_keys(Keys.ENTER)
    time.sleep(3)

    # 欢迎页截图
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "textarea")))
    time.sleep(2)
    driver.save_screenshot(str(OUT / "ui-welcome.png"))
    print("ui-welcome.png OK")

    # 提问并等待回答
    box = driver.find_element(By.CSS_SELECTOR, "textarea")
    box.send_keys("25万预算推荐什么SUV")
    box.send_keys(Keys.ENTER)
    # 等待回答文本出现（等待含"Model Y"或"推荐"的元素，最长90s）
    t0 = time.time()
    while time.time() - t0 < 90:
        body = driver.find_element(By.TAG_NAME, "body").text
        if ("Model Y" in body or "理想" in body) and "正在分析" not in body:
            time.sleep(2)
            break
        time.sleep(2)
    driver.save_screenshot(str(OUT / "ui-chat.png"))
    print("ui-chat.png OK")
finally:
    driver.quit()
