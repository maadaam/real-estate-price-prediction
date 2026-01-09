# extract_features.py файл для извлечения признаков из спаршенных данных
import pandas as pd
import numpy as np
import re

# Список станций метро Санкт-Петербурга (актуальный)
SPB_METRO_STATIONS = [
    "Автово", "Адмиралтейская", "Академическая", "Балтийская", "Беговая", 
    "Бухарестская", "Василеостровская", "Владимирская", "Волковская", 
    "Выборгская", "Горьковская", "Гостиный двор", "Гражданский проспект", 
    "Девяткино", "Достоевская", "Дунайская", "Елизаровская", "Звёздная", 
    "Зенит", "Кировский завод", "Комендантский проспект", "Крестовский остров", 
    "Купчино", "Ладожская", "Ленинский проспект", "Лесная", "Лиговский проспект", 
    "Ломоносовская", "Маяковская", "Международная", "Московская", 
    "Московские ворота", "Нарвская", "Невский проспект", "Новочеркасская", 
    "Обводный канал", "Обухово", "Озерки", "Парк Победы", "Парнас", 
    "Петроградская", "Пионерская", "Площадь Александра Невского 1", 
    "Площадь Александра Невского 2", "Площадь Восстания", "Площадь Ленина", 
    "Площадь Мужества", "Политехническая", "Приморская", "Пролетарская", 
    "Проспект Большевиков", "Проспект Ветеранов", "Проспект Просвещения", 
    "Пушкинская", "Рыбацкое", "Садовая", "Сенная площадь", "Спасская", 
    "Спортивная", "Старая Деревня", "Технологический институт 1", 
    "Технологический институт 2", "Удельная", "Улица Дыбенко", "Фрунзенская", 
    "Чёрная речка", "Чернышевская", "Чкаловская", "Электросила", 
    "Юго-Западная", "Южная", "Шушары", "Путиловская", "Казаковская"
]

# Функция для поиска метро в тексте
def find_metro_in_text(text):
    """Находит все упоминания станций метро в тексте"""
    found_stations = []
    
    # Приводим текст к нижнему регистру для поиска (но сохраняем оригинальное написание)
    text_lower = text.lower()
    
    # Ищем каждую станцию в тексте
    for station in SPB_METRO_STATIONS:
        station_lower = station.lower()
        
        # Проверяем разные варианты написания
        patterns = [
            f"метро {station_lower}",  # "метро московская"
            f"м. {station_lower}",      # "м. московская"
            f"м {station_lower}",       # "м московская"
            f"ст. {station_lower}",     # "ст. московская"
            f"станция {station_lower}", # "станция московская"
        ]
        
        # Также проверяем просто название станции (но с границами слов)
        patterns.append(rf"\b{re.escape(station_lower)}\b")
        
        for pattern in patterns:
            if re.search(pattern, text_lower):
                # Нашли упоминание - добавляем оригинальное название
                found_stations.append(station)
                break  # Не добавляем дубли
    
    return list(set(found_stations))  # Убираем дубликаты

# ----------------------
# 1. Основная функция для извлечения признаков из текста
# ----------------------
def extract_refined_features(text):
    if not isinstance(text, str):
        return pd.Series({})

    res = {}

    # 1. Находим все станции метро в тексте
    metro_stations = find_metro_in_text(text)
    
    # 2. Ищем время до метро
    metro_time_patterns = [
        r"(\d+)(?:[–\-]\d+)?\s*мин(?:\.|ут)?\s*(?:до|от)?\s*(?:метро|м\.|м\s|ст\.|станции)",
        r"(?:до|от)\s*(?:метро|м\.|м\s|ст\.|станции)\s*(\d+)(?:[–\-]\d+)?\s*мин(?:\.|ут)?",
        r"(\d+)(?:[–\-]\d+)?\s*минут",
        r"пешком\s*(\d+)(?:[–\-]\d+)?\s*мин",
    ]
    
    metro_times = []
    for pattern in metro_time_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if match.group(1):
                metro_times.append(int(match.group(1)))
    
    # Заполняем результаты
    if metro_stations:
        # Сортируем для единообразия
        metro_stations_sorted = sorted(metro_stations)
        res['все_метро'] = ", ".join(metro_stations_sorted)
        
        # Берём минимальное время из найденных
        if metro_times:
            res['время_до_ближайшего_мин'] = min(metro_times)
        else:
            # Если время не указано, но метро есть - ставим дефолтное значение
            res['время_до_ближайшего_мин'] = 15
    else:
        res['все_метро'] = "nan"
        res['время_до_ближайшего_мин'] = "nan"

    # --- О КВАРТИРЕ ---
    res['комнат'] = re.search(r"Количество комнат:\s*(.*?)(?=\s*Общая площадь|$)", text)
    res['площадь_общая'] = re.search(r"Общая площадь:\s*([\d.]+)", text)
    res['площадь_кухни'] = re.search(r"Площадь кухни:\s*([\d.]+)", text)
    res['площадь_жилая'] = re.search(r"Жилая площадь:\s*([\d.]+)", text)

    floor_match = re.search(r"Этаж:\s*(\d+)\s*из\s*(\d+)", text)
    res['этаж'] = floor_match.group(1) if floor_match else "nan"
    res['этажность_дома'] = floor_match.group(2) if floor_match else "nan"

    res['тип_комнат'] = re.search(r"Тип комнат:\s*(.*?)(?=\s*(?:Высота|Санузел|Окна|Балкон|$))", text)
    res['высота_потолков'] = re.search(r"Высота потолков:\s*([\d.]+)", text)
    res['санузел'] = re.search(r"Санузел:\s*(.*?)(?=\s*(?:Окна|Ремонт|Техника|$))", text)
    res['окна'] = re.search(r"Окна:\s*(.*?)(?=\s*(?:Ремонт|Способ|Тёплый|$))", text)
    res['ремонт'] = re.search(r"Ремонт:\s*(.*?)(?=\s*(?:Тёплый|Техника|Способ|Вид|Мебель|$))", text)

    # --- МЕБЕЛЬ ---
    match = re.search(r"Мебель:\s*(.+?)(?=\s*$|\n|$)", text, flags=re.IGNORECASE)
    furniture_list = [i.strip() for i in match.group(1).split(",")] if match else []
    res['furniture_list'] = furniture_list

    # --- СПОСОБ ПРОДАЖИ ---
    match = re.search(r"Способ продажи[:\s]*([^\n\.]+)", text, flags=re.IGNORECASE)
    if match:
        res['способ_продажи'] = match.group(1).strip()
    else:
        alt_match = re.search(r"(Прямая продажа|Возможна ипотека|Только наличный расчёт)", text, flags=re.IGNORECASE)
        res['способ_продажи'] = alt_match.group(1).strip() if alt_match else "Н/Д"

    res['вид_сделки'] = re.search(r"Вид сделки:\s*(.*?)(?=\s*(?:Рыночная|Оценка|Проверка|$))", text)

    # --- О ДОМЕ ---
    res['тип_дома'] = re.search(r"Тип дома:\s*([\w-]+)", text)
    res['год_постройки'] = re.search(r"Год постройки:\s*(\d{4})", text)
    res['в_доме'] = re.search(r"В доме:\s*(.*?)(?=\s*(?:Двор|Парковка|Узнать|$))", text)
    res['двор'] = re.search(r"Двор:\s*(.*?)(?=\s*(?:Парковка|Узнать|В доме|$))", text)
    res['парковка'] = re.search(r"Парковка:\s*(.*?)(?=\s*(?:Рассчитайте|Узнать|О доме|$))", text)
    res['лифт_пасс'] = re.search(r"Пассажирский лифт:\s*(\d+)", text)
    res['лифт_груз'] = re.search(r"Грузовой лифт:\s*(\d+)", text)

    # --- Чистка объектов Match ---
    for key in res:
        if hasattr(res[key], 'group') and res[key] is not None:
            res[key] = res[key].group(1).strip()
        elif res[key] is None or res[key] == "nan":
            res[key] = "nan"

    return pd.Series(res)


# ----------------------
# 2. Вспомогательные функции
# ----------------------
def extract_furniture(f_list):
    if not f_list or f_list == []:
        return pd.Series({"кухня":0,"спальные места":0,"шкаф/хранение":0,"техника":0,"диван/кровать":0})
    text = " ".join(f_list) if isinstance(f_list, list) else str(f_list)
    text = text.lower()
    return pd.Series({
        "кухня": int("кухня" in text),
        "спальные места": int("спальные места" in text),
        "шкаф/хранение": int("хранение" in text or "шкаф" in text),
        "техника": int("техника" in text or "холодильник" in text or "посудомоечная" in text or "стиральная" in text),
        "диван/кровать": int("диван" in text or "кровать" in text)
    })

def parse_title(title):
    title = str(title).lower()
    is_apartment = int("апартамент" in title)
    if "студия" in title or "студии" in title:
        rooms = 0.7
    else:
        match = re.search(r"(\d+)\s*-\s*к", title)
        rooms = int(match.group(1)) if match else np.nan
    return pd.Series({"rooms_from_title":rooms, "is_apartment":is_apartment})

def encode_room_type(val):
    val = str(val).lower()
    return pd.Series({
        "изолированные комнаты": int("изолированные" in val),
        "смежные комнаты": int("смежные" in val),
        "тип_комнаты_неизвестен": int(pd.isna(val) or val=="nan")
    })

def toilet_type(val):
    val = str(val).lower()
    return pd.Series({
        "совмещенный санузел": int("совмещенный" in val),
        "раздельный санузел": int("раздельный" in val),
        "неизвестный санузел": int(pd.isna(val) or val=="nan")
    })

def encode_window(val):
    val = str(val).lower()
    return pd.Series({
        "окна во двор": int("во двор" in val),
        "окна на улицу": int("на улицу" in val),
        "окна на солнечную сторону": int("на солнечную сторону" in val),
        "окна неизвестны": int(pd.isna(val) or val=="nan")
    })

def extract_repair(text):
    text = str(text).lower()
    if "дизайнер" in text: return "дизайнерский"
    if "евро" in text: return "евро"
    if "космет" in text: return "косметический"
    if "требует ремонта" in text: return "требует ремонта"
    return "не указано"

def encode_sposob(val):
    val = str(val).lower()
    return pd.Series({
        "способ_свободная": int("свободная" in val),
        "способ_альтернативная": int("альтернативная" in val),
        "способ_ипотека": int("ипотека" in val),
        "способ_переуступка": int("переуступка" in val),
        "способ_договор долевого участия": int("договор долевого участия" in val),
        "способ_неизвестно": int(pd.isna(val) or val=="nan")
    })

def in_flat_encode(text):
    text = str(text).lower()
    return pd.Series({
        "в_доме_газ": int("газ" in text),
        "в_доме_мусоропровод": int("мусоропровод" in text),
        "в_доме_консьерж": int("консьерж" in text),
        "в_доме_неизвестно": int(pd.isna(text) or "nan" in text)
    })

def yard_encode(text):
    text = str(text).lower()
    return pd.Series({
        "детская_площадка": int("детская площадка" in text),
        "спортивная площадка": int("спортивная площадка" in text),
        "закрытая территория": int("закрытая территория" in text),
        "двор_неизвестно": int(pd.isna(text) or "nan" in text)
    })

def encode_parking(text):
    text = str(text).lower()
    return pd.Series({
        "парковка_подземная": int("подземная" in text),
        "парковка_открытая_во_дворе": int("открытая во дворе" in text),
        "парковка_за_шлагбаумом": int("за шлагбаумом" in text),
        "парковка_наземная_многоуровневая": int("наземная многоуровневая" in text),
        "парковка_неизвестно": int(pd.isna(text) or "nan" in text)
    })

# ----------------------
# 3. Основная функция для подготовки всех признаков
# ----------------------
def make_features(df):
    df_final = df.copy()

    # --- извлечение признаков из текста ---
    new_features = df_final['Все_Данные'].apply(extract_refined_features)
    df_final = pd.concat([df_final[['Название','Цена','Ссылка']], new_features], axis=1)

    # --- мебель ---
    furn = df_final["furniture_list"].apply(extract_furniture)
    df_final = pd.concat([df_final.drop(columns="furniture_list"), furn], axis=1)

    # --- признаки из названия ---
    title_features = df_final["Название"].apply(parse_title)
    df_final = pd.concat([df_final, title_features], axis=1)

    # --- числовые колонки ---
    num_cols = ["Цена","время_до_ближайшего_мин","комнат","площадь_общая",
                "площадь_кухни","площадь_жилая","этаж","этажность_дома",
                "высота_потолков","год_постройки","лифт_пасс","лифт_груз"]
    df_final = df_final.replace(["Н/Д","Не указано"], np.nan)
    for col in num_cols:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

    # --- заполнение пропусков ---
    df_final["все_метро"] = df_final["все_метро"].fillna(0)
    df_final["время_до_ближайшего_мин"] = df_final["время_до_ближайшего_мин"].fillna(60)
    df_final[["лифт_пасс","лифт_груз"]] = df_final[["лифт_пасс","лифт_груз"]].fillna(0)
    df_final["парковка"] = df_final["парковка"].fillna(0)
    df_final["комнат"] = df_final["комнат"].fillna(df_final["rooms_from_title"])
    df_final = df_final.drop(columns=["rooms_from_title"])

    # --- кодировки категориальных признаков ---
    df_final["вид_сделки"] = df_final["вид_сделки"].apply(lambda x: 1 if isinstance(x,str) and "возможна ипотека" in x.lower() else 0)
    df_final["ремонт"] = df_final["ремонт"].apply(extract_repair)

    df_final = pd.concat([df_final.drop(columns="тип_комнат"), df_final["тип_комнат"].apply(encode_room_type)], axis=1)
    df_final = pd.concat([df_final.drop(columns="санузел"), df_final["санузел"].apply(toilet_type)], axis=1)
    df_final = pd.concat([df_final.drop(columns="окна"), df_final["окна"].apply(encode_window)], axis=1)
    df_final = pd.concat([df_final.drop(columns="способ_продажи"), df_final["способ_продажи"].apply(encode_sposob)], axis=1)
    df_final = pd.concat([df_final.drop(columns="в_доме"), df_final["в_доме"].apply(in_flat_encode)], axis=1)
    df_final = pd.concat([df_final.drop(columns="двор"), df_final["двор"].apply(yard_encode)], axis=1)
    df_final = pd.concat([df_final.drop(columns="парковка"), df_final["парковка"].apply(encode_parking)], axis=1)

    # --- специальные условия для студий ---
    df_final.loc[(df_final['комнат']==0.7) & (df_final['площадь_кухни'].isna()), 'площадь_кухни'] = 0


    return df_final
