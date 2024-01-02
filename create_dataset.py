# Создадим датасет о пассажирах “Титаника”: catboost.titanic()

import pandas as pd
from catboost.datasets import titanic

# Загрузим данные о пассажирах "Титаника"
train_df, _ = titanic()
titanic_dataset = train_df.copy()

# Заполним пропущенные значения в поле "Age" средним значением
titanic_dataset['Age'].fillna(titanic_dataset['Age'].mean(), inplace=True)

# Создадим новый признак для поля "Sex" с использованием one-hot-encoding 
titanic_dataset = pd.get_dummies(titanic_dataset, columns=['Sex'], drop_first=True)

# Сохраним датасет в CSV файл
titanic_dataset.to_csv("./datasets/dataset_titanic1.csv", index=False)

