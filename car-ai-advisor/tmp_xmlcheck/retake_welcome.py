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
driver = webdriver.Edge(opts)
wait = WebDriverWait(driver, 90)
try:
    driver.get("http://localhost:8501")
    time.sleep(4)
    body0 = driver.find_element(By.TAG_NAME, "body").text
    if "演示站访问验证" in body0:
        pwd = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='password']")))
        pwd.click()
        pwd.send_keys(PWD)
        pwd.send_keys(Keys.ENTER)
    # 门禁通过后才会出现聊天输入框（textarea）；等它出现再稳 3 秒
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "textarea")))
    t0 = time.time()
    while time.time() - t0 < 60:
        body = driver.find_element(By.TAG_NAME, "body").text
        if "演示站访问验证" not in body:
            break
        time.sleep(1)
    time.sleep(3)
    driver.save_screenshot(str(OUT / "ui-welcome.png"))
    print("ui-welcome.png retaken, gate in shot:", "演示站访问验证" in driver.find_element(By.TAG_NAME, "body").text)
finally:
    driver.quit()
