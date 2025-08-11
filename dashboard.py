# FILE: dashboard.py

import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime
import graphviz # Убедитесь, что graphviz установлен: pip install streamlit-graphviz

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(
    page_title="AI Factory Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- КОНСТАНТЫ И КОНФИГУРАЦИЯ ---
STATE_FILE = "output/graph_state.json"
API_LOG_FILE = "output/api_usage_log.json"
API_LIMITS = {
    "gemini-2.5-pro": 100, "gemini-2.5-flash": 250, "gemini-2.5-flash-lite": 1000,
    "gemma-3": 14400, "gemma-3n": 14400, "gemini-embedding-001": 1000,
}

# --- ФУНКЦИИ-ПОМОЩНИКИ ---
# Убираем ttl, чтобы данные всегда были свежими при ручном обновлении
def load_json_data(filepath: str) -> dict | None:
    """
    Функция для загрузки JSON-файла. БЕЗ кэширования.
    """
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
    return None

def generate_graphviz_dot(state_data: dict) -> str:
    """Генерирует DOT-строку для Graphviz на основе состояния графа."""
    dot = graphviz.Digraph(comment='Граф Задач')
    dot.attr('graph', rankdir='TB', splines='ortho', bgcolor='transparent')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')
    dot.attr('edge', fontname='Arial', fontsize='8')

    all_tasks = state_data.get("task_queue", []) + state_data.get("completed_tasks", [])
    current_task_id = state_data.get("current_task", {}).get("task_id")
    
    # Собираем ID всех завершенных задач
    completed_ids = {task['task_id'] for task in state_data.get("completed_tasks", [])}
    
    # Собираем ID задач, которые завершились с ошибкой
    failed_ids = {task['task_id'] for task in state_data.get("completed_tasks", []) if task.get('status') == 'FAILURE'}

    if not all_tasks:
        dot.node("no_tasks", "План задач еще не сгенерирован", fillcolor='lightgrey')
        return dot.source

    # Используем set для хранения уникальных ID задач, чтобы избежать дублирования узлов
    processed_nodes = set()
    for task in all_tasks:
        task_id = task['task_id']
        if task_id in processed_nodes:
            continue
        processed_nodes.add(task_id)

        label = f"<{task_id}<br/><font point-size='8'>{task['agent_name']}</font>>"
        
        # Определение цвета узла
        if task_id == current_task_id:
            color = 'lightblue'
            penwidth = '2.0'
        elif task_id in failed_ids:
            color = 'lightcoral'
            penwidth = '1.0'
        elif task_id in completed_ids:
            color = 'lightgreen'
            penwidth = '1.0'
        else:
            color = 'ivory'
            penwidth = '1.0'

        dot.node(task_id, label, fillcolor=color, penwidth=penwidth)

        # Добавление ребер зависимостей
        for dep_id in task.get("dependencies", []):
            dot.edge(dep_id, task_id)
            
    return dot.source

# --- ГЛАВНЫЙ ИНТЕРФЕЙС ---
st.title("🤖 Панель Мониторинга: Фабрика Аналитики v5.0 (Production Ready)")

# Убираем автоматическое обновление, добавляем кнопку для ручного
st.button("🔄 Обновить данные")

state_data = load_json_data(STATE_FILE)
api_log_data = load_json_data(API_LOG_FILE)

if not state_data:
    st.warning(f"Ожидание запуска основного процесса... Файл состояния '{STATE_FILE}' не найден.", icon="⏳")
else:
    # --- СЕКЦИЯ 1: ОБЩИЙ ПРОГРЕСС ---
    st.header("📊 Общий Прогресс Выполнения")
    
    all_tasks_list = state_data.get("task_queue", []) + state_data.get("completed_tasks", [])
    total_tasks = len(set(t['task_id'] for t in all_tasks_list))
    completed_tasks_count = len(set(t['task_id'] for t in state_data.get("completed_tasks", []) if t.get('status') == 'SUCCESS'))
    progress = completed_tasks_count / total_tasks if total_tasks > 0 else 0
    
    st.progress(progress, text=f"Выполнено {completed_tasks_count} из {total_tasks} уникальных задач")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Задач в Очереди", len(state_data.get("task_queue", [])))
    col2.metric("Задач Выполнено", completed_tasks_count)
    col3.metric("Фактов в Базе Знаний", len(state_data.get("knowledge_base", {})))
    col4.metric("Создано Артефактов", len(state_data.get("artifacts", {})))

    # --- СЕКЦИЯ 2: ВКЛАДКИ С ДАННЫМИ ---
    tab_graph, tab_details, tab_kb, tab_artifacts, tab_api = st.tabs([
        "📈 Граф Задач", "📋 Детализация Задач", "🧠 База Знаний", "📄 Артефакты", "💰 Расходы API"
    ])

    with tab_graph:
        st.subheader("🔗 Визуализация Плана Выполнения")
        try:
            dot_source = generate_graphviz_dot(state_data)
            st.graphviz_chart(dot_source)
        except Exception as e:
            st.error(f"Не удалось построить граф: {e}")

    with tab_details:
        col1, col2 = st.columns(2)
        with col1:
            st.info("🔹 **Текущая Задача**", icon="⚙️")
            current_task = state_data.get("current_task")
            if current_task:
                st.json(current_task, expanded=True)
            else:
                st.write("Нет активной задачи.")

        with col2:
            st.success("🔸 **Последние Выполненные Задачи**", icon="✅")
            completed_tasks = state_data.get("completed_tasks", [])
            if completed_tasks:
                st.json(completed_tasks[-5:][::-1], expanded=True)
            else:
                st.write("Еще нет выполненных задач.")

        with st.expander("Показать всю очередь задач"):
            st.json(state_data.get("task_queue", []))

    with tab_kb:
        st.subheader("🧠 База Знаний")
        kb = state_data.get("knowledge_base", {})
        if kb:
            kb_list = [{"claim_id": k, **v} for k, v in kb.items()]
            df_kb = pd.DataFrame(kb_list)
            st.dataframe(df_kb[["claim_id", "statement", "status", "source_link", "created_at"]])
        else:
            st.info("База знаний пока пуста.")

    with tab_artifacts:
        st.subheader("📄 Сгенерированные Бизнес-Артефакты")
        artifacts = state_data.get("artifacts", {})
        if artifacts:
            for artifact_id, artifact_data in artifacts.items():
                with st.expander(f"**Артефакт:** `{artifact_id}` ({artifact_data.get('title', 'Без названия')})"):
                    st.json(artifact_data, expanded=False)
                    if 'calculations_table_markdown' in artifact_data:
                        st.markdown("---")
                        st.markdown(artifact_data['calculations_table_markdown'])
                    if 'mermaid_diagram' in artifact_data:
                        st.markdown("---")
                        st.graphviz_chart(artifact_data['mermaid_diagram'])
        else:
            st.info("Еще не создано ни одного артефакта.")

    with tab_api:
        st.subheader("💰 Расходы API (за текущие сутки)")
        if api_log_data and "usage" in api_log_data:
            usage_data = api_log_data["usage"]
            
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