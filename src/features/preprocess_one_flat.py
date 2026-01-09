#файл для подготовки данных введенных пользователем для предсказания моделью
import pandas as pd
import numpy as np
import joblib
import os
import re
import warnings
from collections import Counter

class SingleFlatProcessor:
    def __init__(self, artifacts_path="artifacts/preprocessing_artifacts.joblib"):
        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(f"{artifacts_path} не найден")

        artifacts = joblib.load(artifacts_path)

        # ===== базовые артефакты =====
        self.scaler = artifacts["scaler"]
        self.scale_features = artifacts["scale_features"]
        self.log_features = artifacts["log_features"]
        self.feature_columns = artifacts["feature_columns"]
        self.CURRENT_YEAR = artifacts["CURRENT_YEAR"]
        self.categorical_cols = artifacts["categorical_cols"]
        
        # ===== статистики метро =====
        self.metro_freq = artifacts["metro_freq"]
        self.metro_price_dict = artifacts["metro_price_dict"]
        self.global_avg_price = artifacts["global_avg_price"]
        
        # ===== статистики для заполнения пропусков =====
        self.kitchen_ratio_map = {float(k): v for k, v in artifacts["kitchen_ratio_map"].items()}
        self.global_k_ratio = artifacts["global_k_ratio"]
        self.living_ratio_map = {float(k): v for k, v in artifacts["living_ratio_map"].items()}
        self.global_l_ratio = artifacts["global_l_ratio"]
        self.height_median = artifacts["height_median"]

        self.median_type_floor = artifacts["median_type_floor"]
        self.median_type = artifacts["median_type"]
        self.median_floor = artifacts["median_floor"]
        self.global_median_year = artifacts["global_median_year"]

        self.median_rooms_per_bin = artifacts["median_rooms_per_bin"]
        self.global_median_rooms = artifacts["global_median_rooms"]
        
        # ===== бины для категориальных признаков =====
        self.area_bins = artifacts["area_bins"]
        self.height_bins = artifacts["height_bins"]
        self.height_labels = artifacts["height_labels"]
        self.metro_time_bins = artifacts["metro_time_bins"]
        self.metro_time_labels = artifacts["metro_time_labels"]
        self.age_bins = artifacts["age_bins"]
        self.age_labels = artifacts["age_labels"]
        self.rooms_bins = artifacts["rooms_bins"]
        self.rooms_labels = artifacts["rooms_labels"]
        
        # ===== границы выбросов (оставляем только нужные) =====
        self.price_bounds = artifacts["price_bounds"]
        self.ceiling_bounds = artifacts["ceiling_bounds"]

    # ======================================================
    # RAW
    # ======================================================
    def prepare_raw_input(self, user_data):
        df = pd.DataFrame([user_data])

        num_cols = [
            "время_до_ближайшего_мин", "комнат", "площадь_общая",
            "площадь_кухни", "площадь_жилая", "этаж",
            "этажность_дома", "высота_потолков",
            "год_постройки", "лифт_пасс", "лифт_груз"
        ]

        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Заполняем пропуски как в основном файле
        df["время_до_ближайшего_мин"] = df.get("время_до_ближайшего_мин", 60).fillna(60)
        df["лифт_пасс"] = df.get("лифт_пасс", 0).fillna(0)
        df["лифт_груз"] = df.get("лифт_груз", 0).fillna(0)

        return df

    # ======================================================
    # PIPELINE
    # ======================================================
    def process_single_flat(self, user_data):
        df = self.prepare_raw_input(user_data)
        df = self.fill_missing_values(df)
        df = self.process_metro(df)
        df = self.add_extra_features(df)
        df = self.apply_binning(df)
        df = self.create_categorical_features(df)
        df = self.apply_log(df)
        df = self.apply_scaling(df)
        df = self.convert_categorical_to_str(df)
        df = self.ensure_column_order(df)
        return df

    # ======================================================
    # STEPS
    # ======================================================
    def fill_missing_values(self, df):
        df = df.copy()

        # Создаем признак "год_не_указан" ДО заполнения года (как в основном файле)
        df["год_не_указан"] = df["год_постройки"].isna().astype(int)
        
        # Заполняем площадь кухни
        if pd.isna(df.at[0, "площадь_кухни"]):
            r = df.at[0, "комнат"]
            df.at[0, "площадь_кухни"] = df.at[0, "площадь_общая"] * self.kitchen_ratio_map.get(r, self.global_k_ratio)

        # Заполняем жилую площадь
        if pd.isna(df.at[0, "площадь_жилая"]):
            r = df.at[0, "комнат"]
            df.at[0, "площадь_жилая"] = df.at[0, "площадь_общая"] * self.living_ratio_map.get(r, self.global_l_ratio)

        # Заполняем высоту потолков
        if pd.isna(df.at[0, "высота_потолков"]):
            df.at[0, "высота_потолков"] = self.height_median

        # Заполняем год постройки (как в основном файле)
        if pd.isna(df.at[0, "год_постройки"]):
            t = df.get("тип_дома", [None])[0]
            f = df.get("этажность_дома", [None])[0]
            
            # Используем тот же порядок приоритетов
            key = f"{t}_{f}"
            df.at[0, "год_постройки"] = (
                self.median_type_floor.get(key)
                or self.median_type.get(t)
                or self.median_floor.get(f)
                or self.global_median_year
            )

        return df

    def clean_metro_name(self, m):
        m = m.lower()
        m = re.sub(r"от$", "", m)
        m = re.sub(r"[^а-яё\s\-]", "", m)
        return m.strip().title()

    def process_metro(self, df):
        df = df.copy()
        
        # Обработка метро для получения ближайшего метро
        stations_raw = str(df.get("все_метро", [""])[0])
        
        def split_metro(val):
            if val == 0 or pd.isna(val) or str(val).strip() == "":
                return []
            return [self.clean_metro_name(m) for m in str(val).split(",")]
        
        metro_list = split_metro(stations_raw)
        
        # Извлекаем ближайшее метро (первое в списке)
        ближайшее_метро = metro_list[0] if len(metro_list) > 0 else "не указано"
        df["ближайшее_метро"] = ближайшее_метро
        
        # Частотные признаки метро (как в основном файле)
        def get_metro_features(metro_list):
            if not metro_list:
                return 0, 0, 0
            freqs = [self.metro_freq.get(m, 0) for m in metro_list]
            if not freqs:
                return 0, 0, 0
            return len(metro_list), max(freqs), np.mean(freqs)
        
        df["количество_метро"], df["макс_частота_метро"], df["ср_частота_метро"] = get_metro_features(metro_list)
        
        # Признак средней цены по метро (из артефактов)
        df["средняя_цена_по_метро"] = self.metro_price_dict.get(ближайшее_метро, self.global_avg_price)
        
        # Удаляем исходный столбец метро
        df.drop(columns=["все_метро"], inplace=True, errors="ignore")
        
        return df

    def add_extra_features(self, df):
        df = df.copy()

        # Возраст дома (как в основном файле)
        df["возраст_дома"] = self.CURRENT_YEAR - df["год_постройки"]
        df.drop(columns=["год_постройки"], inplace=True, errors="ignore")
        
        # Отношения площадей
        df["кухня_доля"] = df["площадь_кухни"] / df["площадь_общая"]
        df["жилая_доля"] = df["площадь_жилая"] / df["площадь_общая"]
        df["нежилая_доля"] = 1 - df["жилая_доля"]
        
        # Признаки на основе комнат (добавляем комнат_на_100м2 как в основном файле)
        df["площадь_на_комнату"] = df["площадь_общая"] / df["комнат"]
        df["комнат_на_100м2"] = df["комнат"] / df["площадь_общая"] * 100
        
        # Этажные признаки (добавляем средний_этаж как в основном файле)
        df["этаж_относительный"] = df["этаж"] / df["этажность_дома"]
        df["первый_этаж"] = (df["этаж"] == 1).astype(int)
        df["последний_этаж"] = (df["этаж"] == df["этажность_дома"]).astype(int)
        df["средний_этаж"] = ((df["этаж"] > 1) & (df["этаж"] < df["этажность_дома"])).astype(int)
        
        # Бинарные признаки доступности метро
        df["рядом_метро_5мин"] = (df["время_до_ближайшего_мин"] <= 5).astype(int)
        df["рядом_метро_10мин"] = (df["время_до_ближайшего_мин"] <= 10).astype(int)
        
        # Бинарные признаки возраста (как в основном файле)
        df["новостройка"] = (df["возраст_дома"] <= 5).astype(int)
        df["старый_фонд"] = (df["возраст_дома"] >= 70).astype(int)
        
        # Бинарные признаки по комнатам
        df["студия"] = (df["комнат"] == 0.7).astype(int)
        df["большая_квартира"] = (df["комнат"] >= 3).astype(int)
        
        return df

    def create_categorical_features(self, df):
        """Создание категориальных признаков (как в основном файле)"""
        df = df.copy()
        
        # Высотность категория
        if 'этажность_дома' in df.columns:
            df['высотность_категория'] = pd.cut(
                df['этажность_дома'], 
                bins=self.height_bins, 
                labels=self.height_labels
            )
        
        # Доступность метро
        if 'время_до_ближайшего_мин' in df.columns:
            df['доступность_метро'] = pd.cut(
                df['время_до_ближайшего_мин'], 
                bins=self.metro_time_bins, 
                labels=self.metro_time_labels
            )
        
        # Возраст категория
        if 'возраст_дома' in df.columns:
            df['возраст_категория'] = pd.cut(
                df['возраст_дома'], 
                bins=self.age_bins, 
                labels=self.age_labels
            )
        
        # Комнаты категория
        if 'комнат' in df.columns:
            df['комнаты_категория'] = pd.cut(
                df['комнат'], 
                bins=self.rooms_bins, 
                labels=self.rooms_labels
            )
        
        return df

    def apply_binning(self, df):
        df = df.copy()

        # Используем бины площади для заполнения пропущенных комнат
        # (как в основном файле, но с учетом, что bins уже есть в артефактах)
        if pd.isna(df.at[0, "комнат"]):
            # Определяем bin для площади
            area_val = df.at[0, "площадь_общая"]
            bin_assigned = None
            
            # Находим подходящий bin (ручной поиск, так как pd.cut для одного значения сложнее)
            for i in range(len(self.area_bins)-1):
                if self.area_bins[i] <= area_val < self.area_bins[i+1]:
                    bin_assigned = i
                    break
            
            # Используем медианное значение для этого bin
            if bin_assigned in self.median_rooms_per_bin:
                df.at[0, "комнат"] = self.median_rooms_per_bin[bin_assigned]
            else:
                df.at[0, "комнат"] = self.global_median_rooms

        return df

    def apply_log(self, df):
        df = df.copy()
        for col in self.log_features:
            if col in df.columns:
                df[col] = np.log1p(df[col])
        return df

    def apply_scaling(self, df):
        df = df.copy()

        # Создаем DataFrame с ожидаемыми колонками для масштабирования
        expected = [col for col in self.scale_features if col in df.columns]
        
        if expected:
            # Масштабируем только те признаки, которые есть в данных
            X_scaled = self.scaler.transform(df[expected])
            
            # Создаем временный DataFrame с масштабированными значениями
            scaled_df = pd.DataFrame(X_scaled, columns=expected, index=df.index)
            
            # Обновляем исходные значения
            for col in expected:
                df[col] = scaled_df[col]

        return df

    def convert_categorical_to_str(self, df):
        """Преобразуем категориальные признаки в строки для CatBoost"""
        df = df.copy()
        
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        return df

    def ensure_column_order(self, df):
        df = df.copy()

        # Добавляем отсутствующие колонки
        for col in self.feature_columns:
            if col not in df.columns:
                # Для категориальных признаков заполняем строкой "не указано" или "неизвестно"
                if col in self.categorical_cols:
                    # Проверяем тип категориального признака
                    if 'метро' in col.lower():
                        df[col] = "не указано"
                    else:
                        df[col] = "неизвестно"
                else:
                    df[col] = 0

        # Убираем лишние колонки (включая "Название" и "Ссылка" как в основном файле)
        extra_cols = set(df.columns) - set(self.feature_columns)
        if extra_cols:
            df = df.drop(columns=list(extra_cols))

        # Упорядочиваем колонки
        return df[self.feature_columns]