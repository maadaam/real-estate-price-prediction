import os
import csv
import asyncio
import random
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

FILE_NAME = "avito_dump_01.csv"
START_PAGE = 130
PAGES_TO_SCAN = 1000
BASE_URL = "https://www.avito.ru/sankt-peterburg/kvartiry/prodam"
MAX_CONCURRENT_ADS = 5
RETRY_LIMIT = 3
CAPTCHA_WAIT = 15  # секунд

async def load_existing_links(filename):
    links = set()
    if os.path.exists(filename):
        with open(filename, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                links.add(row["Ссылка"])
    return links

async def save_ad(f, title, price, all_text, url):
    writer = csv.writer(f)
    writer.writerow([title, price, all_text, url])
    f.flush()

async def parse_ad(ad_url, context, saved_links):
    if ad_url in saved_links:
        return None

    page = await context.new_page()
    await Stealth().apply_stealth_async(page)
    try:
        for attempt in range(RETRY_LIMIT):
            try:
                await page.bring_to_front()
                await page.goto(ad_url, timeout=45000)
                content = await page.content()
                if "captcha" in page.url or "Проверка" in content:
                    print(f"\n!!! КАПЧА — ждём {CAPTCHA_WAIT} сек !!!")
                    await asyncio.sleep(CAPTCHA_WAIT)
                    continue
                await page.wait_for_selector("h1", timeout=15000)
                break
            except Exception as e:
                print(f"Попытка {attempt+1} на объявлении {ad_url} не удалась: {e}")
                await asyncio.sleep(5)
        else:
            return None

        # Минимальный скролл
        await page.mouse.wheel(0, random.randint(300, 600))
        await asyncio.sleep(random.uniform(0.5, 1.0))

        title = await page.locator("h1").inner_text()
        price_el = page.locator('[itemprop="price"]').first
        price = await price_el.get_attribute("content") if await price_el.count() > 0 else "0"
        all_text = " ".join((await page.locator("body").inner_text()).split())

        return title, price, all_text, ad_url
    finally:
        await page.close()

async def get_listing_links(page):
    for attempt in range(RETRY_LIMIT):
        try:
            await page.bring_to_front()
            await page.wait_for_selector('[data-marker="item-title"]', timeout=60000)
            # скроллим, чтобы подгрузились все объявления
            for i in range(0, 5000, 500):
                await page.evaluate(f"window.scrollTo(0, {i});")
                await asyncio.sleep(random.uniform(0.2, 0.4))

            links = await page.locator('[data-marker="item-title"]').evaluate_all(
                "els => els.map(el => el.href)"
            )
            clean_links = []
            for url in links:
                if url.startswith("http"):
                    clean_links.append(url)
                else:
                    clean_links.append(f"https://www.avito.ru{url}")
            return clean_links

        except Exception as e:
            print(f"Попытка {attempt+1} загрузки листинга не удалась: {e}")
            await asyncio.sleep(5)
    return []

async def run():
    saved_links = await load_existing_links(FILE_NAME)
    print(f"Уже собрано ранее: {len(saved_links)} объявлений")
    collected = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 1024},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/123.0.0.0 Safari/537.36"
        )

        f = open(FILE_NAME, mode="a", encoding="utf-8-sig", newline="")
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["Название", "Цена", "Все_Данные", "Ссылка"])

        for p_num in range(START_PAGE, START_PAGE + PAGES_TO_SCAN):
            print(f"\n=== СТРАНИЦА {p_num} ===")
            try:
                page = await context.new_page()
                await Stealth().apply_stealth_async(page)
                await page.bring_to_front()
                await page.goto(f"{BASE_URL}?p={p_num}", timeout=45000)
                content = await page.content()
                if "captcha" in page.url or "Проверка" in content:
                    print(f"Капча на листинге, ждём {CAPTCHA_WAIT} сек")
                    await asyncio.sleep(CAPTCHA_WAIT)
                    await page.reload()
                    await asyncio.sleep(5)

                links = await get_listing_links(page)
                print(f"Найдено {len(links)} ссылок")

                semaphore = asyncio.Semaphore(MAX_CONCURRENT_ADS)

                async def sem_task(url):
                    async with semaphore:
                        return await parse_ad(url, context, saved_links)

                tasks = [sem_task(url) for url in links]
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    if result:
                        title, price, all_text, ad_url = result
                        await save_ad(f, title, price, all_text, ad_url)
                        saved_links.add(ad_url)
                        collected += 1
                        print(f"Собрано всего: {len(saved_links)}", end="\r")

                await page.close()
                await asyncio.sleep(random.uniform(1.0, 2.0))

            except Exception as e:
                print(f"Ошибка на странице {p_num}: {e}")
                await asyncio.sleep(5)
                continue

        await browser.close()
        f.close()
        print(f"\nГотово. Всего объявлений в датасете: {len(saved_links)}")

if __name__ == "__main__":
    asyncio.run(run())
