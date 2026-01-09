import time
import random
import csv
import os
from playwright.sync_api import sync_playwright
from playwright_stealth import Stealth


FILE_NAME = "avito_dump_02.csv"

# START_PAGE = 250      
PAGES_TO_SCAN = 1000   
BASE_URL = "https://www.avito.ru/sankt-peterburg/kvartiry/prodam"


def load_existing_links(filename):
    """Загружаем уже собранные ссылки"""
    links = set()
    if not os.path.exists(filename):
        return links

    with open(filename, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            links.add(row["Ссылка"])
    return links


def run():
    collected = 0
    saved_links = load_existing_links(FILE_NAME)

    print(f"Уже собрано ранее: {len(saved_links)} объявлений")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            viewport={"width": 1280, "height": 1024},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/123.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        Stealth().apply_stealth_sync(page)

        with open(FILE_NAME, mode="a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["Название", "Цена", "Все_Данные", "Ссылка"])

            for p_num in range(START_PAGE, START_PAGE + PAGES_TO_SCAN):
                print(f"\n=== СТРАНИЦА {p_num} ===")

                try:
                    page.goto(f"{BASE_URL}?p={p_num}", timeout=60000)
                    page.wait_for_selector('[data-marker="item-title"]', timeout=15000)

                    links = [
                        f"https://www.avito.ru{el.get_attribute('href')}"
                        for el in page.locator('[data-marker="item-title"]').all()
                    ]

                    print(f"Найдено {len(links)} ссылок")

                    for ad_url in links:
                        if ad_url in saved_links:
                            continue  # ← уже есть, пропускаем

                        try:
                            page.goto(ad_url, timeout=60000)
                            page.wait_for_selector("h1", timeout=8000)

                            # минимальный «человеческий» скролл
                            page.mouse.wheel(0, random.randint(200, 400))
                            time.sleep(random.uniform(0.6, 1.2))

                            title = page.locator("h1").inner_text()
                            price_el = page.locator('[itemprop="price"]').first
                            price = price_el.get_attribute("content") if price_el.count() > 0 else "0"
                            all_text = " ".join(page.locator("body").inner_text().split())

                            writer.writerow([title, price, all_text, ad_url])
                            f.flush()

                            saved_links.add(ad_url)
                            collected += 1

                            print(f"Собрано всего: {len(saved_links)}", end="\r")

                        except Exception:
                            if "captcha" in page.url:
                                print("\n!!! КАПЧА — ждём 90 сек !!!")
                                time.sleep(90)
                            continue

                        # короткая, но рандомная пауза
                        time.sleep(random.uniform(0.8, 2))

                except Exception as e:
                    print(f"Ошибка на странице {p_num}: {e}")
                    time.sleep(5)
                    continue

        browser.close()
        print(f"\nГотово. Всего объявлений в датасете: {len(saved_links)}")


if __name__ == "__main__":
    run()
