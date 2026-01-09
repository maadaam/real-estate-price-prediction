# файл с обучением модели для предсказания цены квартиры
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import warnings
warnings.filterwarnings('ignore')

# Загружаем обработанные данные
X_train = pd.read_csv("/Users/tomilovdima/avito_flats_ml_project/data/processed/X_train.csv")
y_train = pd.read_csv("/Users/tomilovdima/avito_flats_ml_project/data/processed/y_train.csv").squeeze()
X_test = pd.read_csv("/Users/tomilovdima/avito_flats_ml_project/data/processed/X_test.csv")
y_test = pd.read_csv("/Users/tomilovdima/avito_flats_ml_project/data/processed/y_test.csv").squeeze()


# Загружаем артефакты препроцессинга
artifacts = joblib.load("artefacts/artifacts.joblib")

# Получаем категориальные признаки
categorical_features = artifacts['categorical_cols']
feature_columns = artifacts['feature_columns']

# Убедимся, что порядок колонок совпадает
X_train = X_train[feature_columns]
X_test = X_test[feature_columns]

# ограничиваем максимальную цену 35 млн тк это сильно снижает биас модели (35 млн это 95 процентиль цены всех квартир)
MAX_PRICE = 35_000_000


y_train_clip = np.clip(y_train, 0, MAX_PRICE)
y_test_clip = np.clip(y_test, 0, MAX_PRICE)

# логарифмируем цену
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

def rmse_pct_safe_real(y_true, y_pred, cap=MAX_PRICE): #функция ошибки RMSE
    y_true_clipped = np.clip(y_true, 0, cap)
    y_pred_clipped = np.clip(y_pred, 0, cap)
    rmse_rub = np.sqrt(np.mean((y_true_clipped - y_pred_clipped)**2))
    rmse_pct = rmse_rub / np.mean(y_true_clipped) * 100 if np.mean(y_true_clipped) > 0 else 0
    return rmse_rub, rmse_pct

# Преобразуем категориальные признаки в строковый тип
for col in categorical_features:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype(str).fillna('nan')
        X_test[col] = X_test[col].astype(str).fillna('nan')

# Получаем индексы категориальных признаков для CatBoost
cat_features_indices = []
for col in categorical_features:
    if col in feature_columns:
        cat_features_indices.append(feature_columns.index(col))
    else:
        print(f"Предупреждение: признак {col} не найден в feature_columns")

# Создаем Pool объект для CatBoost
train_pool = Pool(X_train, y_train_log, cat_features=cat_features_indices)

# Параметры для поиска
grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.03, 0.05, 0.07],
    'l2_leaf_reg': [3, 5, 7],
    'iterations': [1000]
}

# Обучаем модель с поиском
model = CatBoostRegressor(
    loss_function='RMSE',
    random_seed=42,
    task_type='CPU',
    early_stopping_rounds=50,
    verbose=100
)

# Запускаем поиск
grid_search_result = model.grid_search(
    grid,
    X=train_pool,
    cv=3,
    partition_random_seed=42,
    verbose=True,
    plot=False
)

print(f"Лучшие параметры: {grid_search_result['params']}")

# Лучшая модель
best_model = model

# Предсказания на train и test
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Трансформируем предсказания обратно из логарифма
y_train_exp = np.expm1(y_train_log)
y_test_exp = np.expm1(y_test_log)
y_train_pred_exp = np.expm1(y_train_pred)
y_test_pred_exp = np.expm1(y_test_pred)

# Вычисляем метрики
train_rmse_rub, train_rmse_pct = rmse_pct_safe_real(y_train_exp, y_train_pred_exp)
test_rmse_rub, test_rmse_pct = rmse_pct_safe_real(y_test_exp, y_test_pred_exp)

print(f"\nРезультаты на тренировочной выборке:")
print(f"RMSE (руб): {train_rmse_rub:,.0f}")
print(f"RMSE (%): {train_rmse_pct:.2f}%")

print(f"\nРезультаты на тестовой выборке:")
print(f"RMSE (руб): {test_rmse_rub:,.0f}")
print(f"RMSE (%): {test_rmse_pct:.2f}%")

# Сохраняем модель
joblib.dump(best_model, "artefacts/catboost_model.joblib")
print(f"\nМодель сохранена в 'artefacts/catboost_model.joblib'")

# Также сохраняем параметры для inference
inference_artifacts = {
    'model': best_model,
    'categorical_cols': categorical_features,
    'feature_columns': feature_columns,
    'cat_features_indices': cat_features_indices,
    'best_params': grid_search_result['params']
}
joblib.dump(inference_artifacts, "artefacts/inference_artifacts.joblib")