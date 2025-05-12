# Poem Generator
Лёгковесный сервис для генерации поэтических продолжений с помощью Seq2Seq-моделей из HuggingFace Transformers. Предоставляет REST API и простой веб-интерфейс на Bootstrap.

## 🔧 Установка

1. **Клонируйте репозиторий**  
   ```bash
   git clone git@github.com:KsuZavyalova/poem_generator.git
   cd poem_generator
2. **Скачайте модели по ссылке**  
https://drive.google.com/drive/folders/1NlOr0GRGB3UwLl2_FOyVGOxSVxn0A-9x?usp=sharing

4. **Создайте и активируйте виртуальное окружение**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Window
5. **Установите зависимости**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt

## 📁 Структура проекта

```
├── app.py
├── length_profiles.json
├── length_profiles_MLE.json
├── reward.py
├── requirements.txt
├── templates/
│   └── index.html
├── epoch-3/
│   └── ...
├── best_model_optuna_2/
│   └── ...
└── README.md
```


## ⚙️ Конфигурация

* Разместите директории моделей (`epoch-3/`, `best_model_optuna_2/`) рядом с `app.py`.
* Файлы `length_profiles.json` и `length_profiles_MLE.json` задают диапазоны длины и оптимальные настройки семплинга.
* Порт и хост можно указать через параметры Uvicorn при запуске.

---

## ▶️ Запуск

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

* Веб-интерфейс: [http://localhost:8000](http://localhost:8000)
* Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
* ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🌐 Веб-интерфейс

1. Откройте [http://localhost:8000](http://localhost:8000).
2. Введите исходный текст и желаемую длину продолжения.
3. Нажмите **Сгенерировать**.
4. Результаты отобразятся в виде карточек со временем выполнения.
