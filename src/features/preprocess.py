# preprocess.py файл с future engineering, обработкой признаков (масштабирование, удаление выбросов)
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from extract_features import make_features 
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from collections import Counter

output_dir = "/Users/tomilovdima/avito_flats_ml_project/data/processed"

# создаем папку, если её нет
import os
os.makedirs(output_dir, exist_ok=True)


# Загружаем сырые данные и извлекаем признаки

df_raw = pd.read_csv("/Users/tomilovdima/avito_flats_ml_project/data/raw/flats1.csv")
df_all = make_features(df_raw)


# Удаляем дубликаты

dedupe_cols = [
    "время_до_ближайшего_мин", "комнат", "площадь_общая", "площадь_кухни", "площадь_жилая",
    "этаж", "этажность_дома", "детская_площадка", "спортивная площадка", "закрытая территория",
    "двор_неизвестно", "парковка_подземная", "парковка_открытая_во_дворе", "парковка_за_шлагбаумом",
    "парковка_наземная_многоуровневая", "парковка_неизвестно"
]
df_all = df_all.drop_duplicates(subset=dedupe_cols, keep='first').reset_index(drop=True)
df_all.to_csv("/Users/tomilovdima/avito_flats_ml_project/data/extr_data.csv", index=False)


# Разделяем на train и test

df_train, df_test = train_test_split(df_all, test_size=0.2, random_state=42, shuffle=True)


# Заполняем пропуски площади кухни и жилой площади

train_ratios = df_train.copy()
train_ratios["k_ratio"] = train_ratios["площадь_кухни"] / train_ratios["площадь_общая"]
kitchen_ratio_map = train_ratios.groupby('комнат')['k_ratio'].median().to_dict()
global_k_ratio = train_ratios['k_ratio'].median()

def fill_kitchen(row, ratio_map, global_ratio):
    if pd.isna(row["площадь_кухни"]):
        ratio = ratio_map.get(row['комнат'], global_ratio)
        return row['площадь_общая'] * ratio
    return row["площадь_кухни"]

df_train['площадь_кухни'] = df_train.apply(fill_kitchen, axis=1, args=(kitchen_ratio_map, global_k_ratio))
df_test['площадь_кухни'] = df_test.apply(fill_kitchen, axis=1, args=(kitchen_ratio_map, global_k_ratio))

# Жилая площадь
train_ratios['l_ratio'] = train_ratios['площадь_жилая'] / train_ratios['площадь_общая']
living_ratio_map = train_ratios.groupby('комнат')['l_ratio'].median().to_dict()
global_l_ratio = train_ratios['l_ratio'].median()

def fill_living(row, ratio_map, global_ratio):
    if pd.isna(row["площадь_жилая"]):
        ratio = ratio_map.get(row['комнат'], global_ratio)
        return row['площадь_общая'] * ratio
    return row["площадь_жилая"]

df_train['площадь_жилая'] = df_train.apply(fill_living, axis=1, args=(living_ratio_map, global_l_ratio))
df_test['площадь_жилая'] = df_test.apply(fill_living, axis=1, args=(living_ratio_map, global_l_ratio))


# Высота потолков и год постройки

heigh_med = df_train["высота_потолков"].median()
df_train["высота_потолков"] = df_train["высота_потолков"].fillna(heigh_med)
df_test["высота_потолков"] = df_test["высота_потолков"].fillna(heigh_med)

df_train["год_не_указан"] = df_train["год_постройки"].isna().astype(int)
df_test["год_не_указан"] = df_test["год_постройки"].isna().astype(int)

median_type_floor = df_train.groupby(["тип_дома","этажность_дома"])["год_постройки"].median()
median_type = df_train.groupby("тип_дома")["год_постройки"].median()
median_floor = df_train.groupby("этажность_дома")["год_постройки"].median()
global_median_year = df_train["год_постройки"].median()

for df in [df_train, df_test]:
    mask = df["год_постройки"].isna()
    df.loc[mask,"год_постройки"] = pd.to_numeric(
        df.loc[mask].set_index(["тип_дома","этажность_дома"]).index.map(median_type_floor),
        errors="coerce"
    )
    mask = df["год_постройки"].isna()
    df.loc[mask,"год_постройки"] = pd.to_numeric(
        df.loc[mask,"тип_дома"].map(median_type),
        errors="coerce"
    )
    mask = df["год_постройки"].isna()
    df.loc[mask,"год_постройки"] = pd.to_numeric(
        df.loc[mask,"этажность_дома"].map(median_floor),
        errors="coerce"
    )
    df["год_постройки"] = df["год_постройки"].fillna(global_median_year).astype("float64")


# Метро - извлекаем ближайшее метро и частотные признаки
def clean_metro_name(m):
    m = m.lower()
    m = re.sub(r"от$", "", m)
    m = re.sub(r"[^а-яё\s\-]", "", m)
    return m.strip().title()

def split_metro(val):
    if val == 0 or pd.isna(val):
        return []
    return [clean_metro_name(m) for m in str(val).split(",")]

df_train["metro_list"] = df_train["все_метро"].apply(split_metro)
# Извлекаем ближайшее метро (первое в списке)
df_train["ближайшее_метро"] = df_train["metro_list"].apply(lambda x: x[0] if len(x) > 0 else "не указано")

df_test["metro_list"] = df_test["все_метро"].apply(split_metro)
df_test["ближайшее_метро"] = df_test["metro_list"].apply(lambda x: x[0] if len(x) > 0 else "не указано")

# Частотные признаки метро (без использования цены)
all_metros = [metro for sublist in df_train["metro_list"] for metro in sublist]
metro_freq = Counter(all_metros)

def get_metro_features(metro_list):
    if not metro_list:
        return 0, 0, 0
    freqs = [metro_freq.get(m, 0) for m in metro_list]
    if not freqs:
        return 0, 0, 0
    return len(metro_list), max(freqs), np.mean(freqs)

df_train["количество_метро"] = df_train["metro_list"].apply(lambda x: get_metro_features(x)[0])
df_train["макс_частота_метро"] = df_train["metro_list"].apply(lambda x: get_metro_features(x)[1])
df_train["ср_частота_метро"] = df_train["metro_list"].apply(lambda x: get_metro_features(x)[2])

df_test["количество_метро"] = df_test["metro_list"].apply(lambda x: get_metro_features(x)[0])
df_test["макс_частота_метро"] = df_test["metro_list"].apply(lambda x: get_metro_features(x)[1])
df_test["ср_частота_метро"] = df_test["metro_list"].apply(lambda x: get_metro_features(x)[2])


#добавляем признак средней цены по метро

# Вычисляем среднюю цену по ближайшему метро только на тренировочных данных
metro_price_stats = df_train.groupby("ближайшее_метро")["Цена"].agg(['mean', 'count']).reset_index()
metro_price_stats.columns = ["ближайшее_метро", "средняя_цена_по_метро", "count_цена_по_метро"]

# Глобальная средняя цена на трейне (на случай, если метро нет в трейне)
global_avg_price = df_train["Цена"].mean()


# Применяем к трейну
df_train = df_train.merge(metro_price_stats[["ближайшее_метро", "средняя_цена_по_метро"]], 
                          on="ближайшее_метро", how="left")

# Применяем к тесту - используем статистики с трейна
df_test = df_test.merge(metro_price_stats[["ближайшее_метро", "средняя_цена_по_метро"]], 
                        on="ближайшее_метро", how="left")

# Заполняем пропуски глобальной средней ценой
df_train["средняя_цена_по_метро"] = df_train["средняя_цена_по_метро"].fillna(global_avg_price)
df_test["средняя_цена_по_метро"] = df_test["средняя_цена_по_метро"].fillna(global_avg_price)


# Удаляем временные столбцы
df_train = df_train.drop(columns=["все_метро", "metro_list"])
df_test = df_test.drop(columns=["все_метро", "metro_list"])

# ----------------------
# 7. Бины по общей площади для заполнения пропусков комнат
# ----------------------
BINS = [0,30,45,60,75,90,120,150,200,np.inf]
LABELS = range(len(BINS)-1)

for df in [df_train, df_test]:
    df['area_bin'] = pd.cut(df['площадь_общая'], bins=BINS, labels=LABELS)

median_rooms_per_bin = df_train.groupby('area_bin', observed=False)['комнат'].median()
global_median_rooms = df_train['комнат'].median()

df_train.loc[df_train['комнат'].isna(), 'комнат'] = df_train.loc[df_train['комнат'].isna(), 'area_bin'].map(median_rooms_per_bin)
df_test.loc[df_test['комнат'].isna(), 'комнат'] = df_test.loc[df_test['комнат'].isna(), 'area_bin'].map(median_rooms_per_bin)
df_train['комнат'] = df_train['комнат'].fillna(global_median_rooms)
df_test['комнат']  = df_test['комнат'].fillna(global_median_rooms)

df_train.drop(columns='area_bin', inplace=True)
df_test.drop(columns='area_bin', inplace=True)


# Дополнительные признаки


CURRENT_YEAR = 2026

# --- Возраст дома ---
df_train["возраст_дома"] = CURRENT_YEAR - df_train["год_постройки"]
df_test["возраст_дома"] = CURRENT_YEAR - df_test["год_постройки"]

# Удаляем уже не нужный год постройки
df_train.drop(columns="год_постройки", inplace=True)
df_test.drop(columns="год_постройки", inplace=True)


# Отношения площадей

for df in [df_train, df_test]:
    df['кухня_доля'] = df['площадь_кухни'] / df['площадь_общая']
    df['жилая_доля'] = df['площадь_жилая'] / df['площадь_общая']
    df['нежилая_доля'] = 1 - df['жилая_доля']


# Признаки на основе комнат

for df in [df_train, df_test]:
    df['площадь_на_комнату'] = df['площадь_общая'] / df['комнат']
    df['комнат_на_100м2'] = df['комнат'] / df['площадь_общая'] * 100


# Этажные признаки

for df in [df_train, df_test]:
    df['этаж_относительный'] = df['этаж'] / df['этажность_дома']
    df['первый_этаж'] = (df['этаж'] == 1).astype(int)
    df['последний_этаж'] = (df['этаж'] == df['этажность_дома']).astype(int)
    df['средний_этаж'] = ((df['этаж'] > 1) & (df['этаж'] < df['этажность_дома'])).astype(int)  # Из старого файла


# Категории высотности дома (bins) - как категориальный признак для CatBoost

высотность_bins = [0, 5, 9, 16, 25, 100]
высотность_labels = ['малоэтажка', 'хрущевка', 'панелька', 'высотка', 'небоскреб']

for df in [df_train, df_test]:
    df['высотность_категория'] = pd.cut(df['этажность_дома'], bins=высотность_bins, labels=высотность_labels)


# Доступность метро - как категориальный признак для CatBoost

доступность_bins = [0, 5, 10, 15, 20, 30, 60, 999]
доступность_labels = ['до_5мин', '5_10мин', '10_15мин', '15_20мин', '20_30мин', '30_60мин', 'далеко']

for df in [df_train, df_test]:
    df['доступность_метро'] = pd.cut(df['время_до_ближайшего_мин'], bins=доступность_bins, labels=доступность_labels)
    df['рядом_метро_5мин'] = (df['время_до_ближайшего_мин'] <= 5).astype(int)
    df['рядом_метро_10мин'] = (df['время_до_ближайшего_мин'] <= 10).astype(int)


# Категории возраста дома - как категориальный признак для CatBoost

возраст_bins = [0, 5, 10, 20, 40, 70, 100, 200]
возраст_labels = ['новостройка', 'свежий', 'современный', 'средний', 'советский', 'сталинка', 'дореволюционный']

for df in [df_train, df_test]:
    df['возраст_категория'] = pd.cut(df['возраст_дома'], bins=возраст_bins, labels=возраст_labels)
    df['новостройка'] = (df['возраст_дома'] <= 5).astype(int)
    df['старый_фонд'] = (df['возраст_дома'] >= 70).astype(int)


# Категории по комнатам - как категориальный признак для CatBoost

комнаты_bins = [0, 0.8, 1.5, 2.5, 3.5, 5, 10, 100]
комнаты_labels = ['студия', '1к', '2к', '3к', '4к', '5к+', 'особняк']

for df in [df_train, df_test]:
    df['комнаты_категория'] = pd.cut(df['комнат'], bins=комнаты_bins, labels=комнаты_labels)
    df['студия'] = (df['комнат'] == 0.7).astype(int)
    df['большая_квартира'] = (df['комнат'] >= 3).astype(int)


# Фильтрация выбросов

# Ограничиваем цену кваритры 1-й и 99-й перцентилем
# price_low, price_high = df_train["Цена"].quantile([0.01, 0.99])
# df_train = df_train[(df_train["Цена"] >= price_low) & (df_train["Цена"] <= price_high)]
price_low, price_high = 700_000, 1_300_000_000 #огранициваем просто маленьким значением, большое клипается в файле обучения модели
df_train = df_train[df_train["Цена"].between(price_low, price_high)]


# Ограничиваем общую площадь квартир 1-й и 99-й перцентилем
# area_low, area_high = df_train["площадь_общая"].quantile([0.01, 0.99])
# df_train = df_train[(df_train["площадь_общая"] >= area_low) & (df_train["площадь_общая"] <= area_high)]

# Физически возможные значения потолков
MIN_CEILING = 1.9
MAX_CEILING = 3

df_train = df_train[(df_train['высота_потолков'] >= MIN_CEILING) & (df_train['высота_потолков'] <= MAX_CEILING)]
df_test = df_test[(df_test['высота_потолков'] >= MIN_CEILING) & (df_test['высота_потолков'] <= MAX_CEILING)]

# Ограничение площади кухни
# df_train = df_train[(df_train['площадь_кухни'] <= 30) & (df_train['площадь_кухни'] / df_train['площадь_общая'] <= 0.3)]
# df_test = df_test[(df_test['площадь_кухни'] <= 30) & (df_test['площадь_кухни'] / df_test['площадь_общая'] <= 0.3)]

print(f"После фильтрации выбросов:")
print(f"  Размер df_train: {df_train.shape}")
print(f"  Размер df_test: {df_test.shape}")


# Логарифмирование признаков (только числовых)

log_features = ["площадь_общая", "площадь_кухни", "площадь_жилая", "средняя_цена_по_метро"]
for df in [df_train, df_test]:
    for col in log_features:
        if col in df.columns:
            df[col] = np.log1p(df[col])


# Масштабирование числовых признаков

scale_features = [
    "время_до_ближайшего_мин", "комнат", "площадь_общая", "площадь_кухни",
    "площадь_жилая", "этаж", "этажность_дома", "высота_потолков",
    "возраст_дома", "средняя_цена_по_метро", "количество_метро",
    "макс_частота_метро", "ср_частота_метро"
]

scaler = StandardScaler()
df_train[scale_features] = scaler.fit_transform(df_train[scale_features])
df_test[scale_features] = scaler.transform(df_test[scale_features])


# Категориальные признаки для CatBoost

categorical_cols = [
    'тип_дома',
    'ближайшее_метро',
    'ремонт',
    'высотность_категория',
    'доступность_метро',
    'возраст_категория',
    'комнаты_категория'
]

# удаляем столбцы "Название" и "Ссылка" перед разделением на X и y
df_train = df_train.drop(columns=["Название", "Ссылка"], errors='ignore')
df_test = df_test.drop(columns=["Название", "Ссылка"], errors='ignore')

# разделяем X и y
X_train = df_train.drop(columns=["Цена"])
y_train = df_train["Цена"]

X_test = df_test.drop(columns=["Цена"])
y_test = df_test["Цена"]


# сохраняем данные 
X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)


# Сохраняем артефакты препроцессинга

# Преобразуем median_type_floor чтобы ключи были строками
median_type_floor_dict = {}
for idx, value in median_type_floor.items():
    if isinstance(idx, tuple):
        key = f"{idx[0]}_{idx[1]}"
    else:
        key = str(idx)
    median_type_floor_dict[key] = float(value)

# Сохраняем статистики по метро для обработки новых данных
metro_price_dict = metro_price_stats.set_index("ближайшее_метро")["средняя_цена_по_метро"].to_dict()

# Собираем ВСЕ статистики в один словарь
artifacts = {
    # 1. Медианы для заполнения пропусков
    "kitchen_ratio_map": kitchen_ratio_map,
    "global_k_ratio": float(global_k_ratio),
    "living_ratio_map": living_ratio_map,
    "global_l_ratio": float(global_l_ratio),
    
    # 2. Медианы высоты потолков
    "height_median": float(heigh_med),
    
    # 3. Медианы года постройки (используем преобразованный словарь)
    "median_type_floor": median_type_floor_dict,  # уже преобразован
    "median_type": median_type.to_dict(),
    "median_floor": median_floor.to_dict(),
    "global_median_year": float(global_median_year),
    
    # 4. Частоты метро (без использования цены)
    "metro_freq": dict(metro_freq),
    
    # 5. Медианы комнат по бинам площади
    "median_rooms_per_bin": median_rooms_per_bin.to_dict(),
    "global_median_rooms": float(global_median_rooms),
    
    # 6. Границы для фильтрации выбросов
    "price_bounds": {
        "low": float(price_low),
        "high": float(price_high)
    },
    # "area_bounds": {
    #     "low": float(area_low),
    #     "high": float(area_high)
    # },
    "ceiling_bounds": {
        "min": MIN_CEILING,
        "max": MAX_CEILING
    },
    
    # 7. Бинаризация (границы бинов для заполнения пропусков)
    "area_bins": BINS,
    
    # 8. Бинаризация категорий
    "height_bins": высотность_bins,
    "height_labels": высотность_labels,
    "metro_time_bins": доступность_bins,
    "metro_time_labels": доступность_labels,
    "age_bins": возраст_bins,
    "age_labels": возраст_labels,
    "rooms_bins": комнаты_bins,
    "rooms_labels": комнаты_labels,
    
    # 9. Масштабирование (scaler)
    "scaler": scaler,
    
    # 10. Текущий год для вычисления возраста
    "CURRENT_YEAR": CURRENT_YEAR,
    
    # 11. Какие фичи логарифмируются
    "log_features": log_features,
    
    # 12. Какие фичи масштабируются
    "scale_features": scale_features,
    
    # 13. Категориальные колонки (для CatBoost)
    "categorical_cols": categorical_cols,
    
    # 14. Порядок колонок 
    "feature_columns": list(X_train.columns),
    
    # 15. Статистики по метро
    "metro_price_dict": metro_price_dict,
    "global_avg_price": float(global_avg_price)
}

# Сохраняем артефакты
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

joblib.dump(artifacts, os.path.join(ARTIFACTS_DIR, "preprocessing_artifacts.joblib"))

