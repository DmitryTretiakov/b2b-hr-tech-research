# dashboard.py
import streamlit as st
import json
import os
import pandas as pd
import time
from datetime import datetime

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(
    page_title="AI Factory Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- КОНСТАНТЫ И КОНФИГУРАЦИЯ ---
STATE_FILE = "output/graph_state.json"
API_LOG_FILE = "output/api_usage_log.json"
REFRESH_INTERVAL_SECONDS = 5
API_LIMITS = {
    "gemini-2.5-pro": 100, "gemini-2.5-flash": 250, "gemini-2.5-flash-lite": 1000,
    "gemma-3": 14400, "gemma-3n": 14400, "gemini-embedding-001": 1000,
}

# --- ФУНКЦИИ-ПОМОЩНИКИ ---
@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_json_data(filepath: str) -> dict | None:
    """Кэшированная функция для загрузки JSON-файла."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None # Файл может быть в процессе записи
    return None

# --- ГЛАВНЫЙ ИНТЕРФЕЙС ---
st.title("🤖 Панель Мониторинга: Фабрика Аналитики v4.2")
st.caption(f"Страница автоматически обновляется каждые {REFRESH_INTERVAL_SECONDS} секунд. Последнее обновление: {datetime.now().strftime('%H:%M:%S')}")

# Создаем placeholder, который будем обновлять
placeholder = st.empty()

while True:
    state_data = load_json_data(STATE_FILE)
    api_log_data = load_json_data(API_LOG_FILE)

    if not state_data:
        with placeholder.container():
            st.warning(f"Ожидание запуска основного процесса... Файл состояния '{STATE_FILE}' не найден.", icon="⏳")
        time.sleep(REFRESH_INTERVAL_SECONDS)
        continue

    with placeholder.container():
        # --- СЕКЦИЯ 1: ОБЩИЙ ПРОГРЕСС ---
        st.header("📊 Общий Прогресс Выполнения")
        
        total_tasks = len(state_data.get("completed_tasks", [])) + len(state_data.get("task_queue", []))
        completed_tasks_count = len(state_data.get("completed_tasks", []))
        progress = completed_tasks_count / total_tasks if total_tasks > 0 else 0
        
        st.progress(progress, text=f"Выполнено {completed_tasks_count} из {total_tasks} задач")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Задач в Очереди", len(state_data.get("task_queue", [])))
        col2.metric("Задач Выполнено", completed_tasks_count)
        col3.metric("Фактов в Базе Знаний", len(state_data.get("knowledge_base", {})))
        col4.metric("Создано Артефактов", len(state_data.get("artifacts", {})))

        # --- СЕКЦИЯ 2: ДЕТАЛИЗАЦИЯ ЗАДАЧ ---
        st.subheader("📋 Детализация Задач")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("🔹 **Текущая Задача**", icon="⚙️")
            current_task = state_data.get("current_task")
            if current_task:
                st.json(current_task, expanded=True)
            else:
                st.write("Нет активной задачи (возможно, идет переход между узлами).")

        with col2:
            st.success("🔸 **Последние Выполненные Задачи**", icon="✅")
            completed_tasks = state_data.get("completed_tasks", [])
            if completed_tasks:
                # Показываем последние 5 задач в обратном порядке
                st.json(completed_tasks[-5:][::-1], expanded=True)
            else:
                st.write("Еще нет выполненных задач.")

        with st.expander("Показать всю очередь задач"):
            st.json(state_data.get("task_queue", []))

        # --- СЕКЦИЯ 3: ВКЛАДКИ С ДАННЫМИ ---
        tab1, tab2, tab3 = st.tabs(["База Знаний (Knowledge Base)", "Сгенерированные Артефакты", "Расходы API"])

        with tab1:
            st.subheader("🧠 База Знаний")
            kb = state_data.get("knowledge_base", {})
            if kb:
                df_kb = pd.DataFrame.from_dict(kb, orient="index")
                st.dataframe(df_kb[["statement", "status", "source_link", "created_at"]])
            else:
                st.info("База знаний пока пуста.")

        with tab2:
            st.subheader("📄 Сгенерированные Бизнес-Артефакты")
            artifacts = state_data.get("artifacts", {})
            if artifacts:
                for artifact_id, artifact_data in artifacts.items():
                    with st.expander(f"**Артефакт:** `{artifact_id}` ({artifact_data.get('title', 'Без названия')})"):
                        st.json(artifact_data, expanded=False)
                        # Если есть Markdown, рендерим его
                        if 'calculations_table_markdown' in artifact_data:
                            st.markdown("---")
                            st.markdown(artifact_data['calculations_table_markdown'])
            else:
                st.info("Еще не создано ни одного артефакта.")

        with tab3:
            st.subheader("💰 Расходы API (за текущие сутки)")
            if api_log_data and "usage" in api_log_data:
                usage_data = api_log_data["usage"]
                
                # Подготовка данных для графика
                chart_data = []
                for model, count in usage_data.items():
                    limit = API_LIMITS.get(model, 0)
                    chart_data.append({
                        "Модель": model,
                        "Использовано": count,
                        "Лимит": limit,
                        "% от лимита": (count / limit * 100) if limit > 0 else 0
                    })
                
                if chart_data:
                    df_chart = pd.DataFrame(chart_data)
                    st.data_editor(
                        df_chart,
                        column_config={
                            "% от лимита": st.column_config.ProgressColumn(
                                "Прогресс",
                                format="%d%%",
                                min_value=0,
                                max_value=100,
                            ),
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("Данные об использовании API пока отсутствуют.")
            else:
                st.info("Лог использования API не найден или пуст.")

    # --- ЛОГИКА ОБНОВЛЕНИЯ ---
    time.sleep(REFRESH_INTERVAL_SECONDS)
    st.rerun()