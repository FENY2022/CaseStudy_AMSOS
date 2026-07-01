"""
Model Manager — Ollama model selection, deep analysis, recommendation with explanations.
"""

import streamlit as st
import pandas as pd

MODEL_KEY = 'selected_model'


def get_ollama_models():
    """Fetch available Ollama models with metadata."""
    try:
        import ollama
        result = ollama.list()
        models = []
        for m in result.models:
            family = m.details.family if m.details and m.details.family else ''
            if family not in ('nomic-bert', ''):
                models.append({
                    'name': m.model,
                    'size': _format_size(m.size) if hasattr(m, 'size') and m.size else 'Unknown',
                    'modified': str(m.modified_at)[:19] if hasattr(m, 'modified_at') and m.modified_at else '',
                })
        return models
    except Exception:
        return []


def get_model_names():
    """Get just the model name strings, with nomic-embed-text excluded."""
    models = get_ollama_models()
    names = [m['name'] for m in models]
    return [n for n in names if 'nomic-embed' not in n]


def _format_size(size_bytes):
    """Format byte size to human-readable."""
    if not size_bytes:
        return 'Unknown'
    size = float(size_bytes)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if size < 1024:
            return f'{size:.1f} {unit}'
        size /= 1024
    return f'{size:.1f} PB'


def query_llm(model_name, system_prompt, user_prompt, temperature=0.3):
    """Query an Ollama model and return the response text."""
    try:
        import ollama
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            options={'temperature': temperature},
        )
        return response['message']['content'].strip()
    except Exception:
        return None


def render_model_selector():
    """Render the model selection dropdown in sidebar."""
    available = get_model_names()
    if not available:
        st.sidebar.markdown(
            '<div style="padding:0.5rem;font-size:0.75rem;color:#DC2626;'
            'background:#FEF2F2;border-radius:6px;">'
            'Ollama not running. Install: <b>ollama serve</b></div>',
            unsafe_allow_html=True,
        )
        st.session_state[MODEL_KEY] = None
        return

    current = st.session_state.get(MODEL_KEY)
    if current not in available:
        current = available[0]
        st.session_state[MODEL_KEY] = current

    idx = available.index(current) if current in available else 0
    selected = st.sidebar.selectbox(
        'AI Model', available, index=idx, key='model_selector',
        format_func=lambda x: x.split(':')[0] if ':' in x else x,
    )
    st.session_state[MODEL_KEY] = selected


def get_selected_model():
    """Get the currently selected model name."""
    return st.session_state.get(MODEL_KEY)


def generate_deep_analysis(model_name, asset_record):
    """Generate deep analysis for a single asset using the selected model."""
    prop = asset_record.get('propertyNumber', 'Unknown')
    equip = asset_record.get('equipment_type', 'Unknown')
    user = asset_record.get('actual_user', 'Unknown')
    employee = asset_record.get('employee', 'Unknown')
    office = asset_record.get('office', 'Unknown')
    year = asset_record.get('year_acquired', 'Unknown')
    shelf = asset_record.get('shelf_life', 'Unknown')
    repairs = asset_record.get('repairs_count', 0)
    health = asset_record.get('health_score', 50)
    risk = asset_record.get('risk_label', 'Unknown')
    status = asset_record.get('current_status', 'Operational')
    issues = ', '.join(asset_record.get('issues_found', [])) or 'None detected'
    recurring = ', '.join(asset_record.get('recurring_issues', {}).keys()) or 'None'
    remarks = asset_record.get('remarks_history', [])
    valid_remarks = [r for r in remarks if pd.notna(r) and str(r).strip() not in ('', 'No remarks recorded')]
    remarks_text = '; '.join(valid_remarks[-5:]) if valid_remarks else 'No repair remarks available.'

    system_prompt = (
        "You are an expert ICT asset management analyst for a government agency. "
        "Analyze the asset data provided and give a concise but thorough assessment. "
        "Structure your response in exactly 4 sections with clear headings:\n\n"
        "## Asset Overview\n(1-2 sentences summarizing the asset and its current state)\n\n"
        "## Deep Analysis\n(2-3 sentences analyzing the repair history, age, and risk factors)\n\n"
        "## Recommendation\n(1-2 sentences giving a specific actionable recommendation)\n\n"
        "## Why\n(1-2 sentences explaining the reasoning behind your recommendation)"
    )

    user_prompt = (
        f"Property No: {prop}\n"
        f"Equipment: {equip}\n"
        f"Assigned To: {employee} ({user})\n"
        f"Office: {office}\n"
        f"Year Acquired: {year}\n"
        f"Shelf Life: {shelf}\n"
        f"Total Repairs: {repairs}\n"
        f"Current Status: {status}\n"
        f"AI Health Score: {health}/100\n"
        f"AI Risk Level: {risk}\n"
        f"Issues Detected: {issues}\n"
        f"Recurring Issues: {recurring}\n"
        f"Recent Repair Remarks: {remarks_text}"
    )

    result = query_llm(model_name, system_prompt, user_prompt, temperature=0.2)
    return result


def generate_procurement_recommendation(model_name, context, budget_info):
    """Generate procurement recommendation with explanation using the selected model."""
    selected_items = context.get('items', [])
    emp_count = context.get('total_employees', 0)
    without_pc = context.get('without_computer', 0)
    offices_needing = context.get('offices_needing', 0)
    required_budget = context.get('required_budget', 0)
    available_budget = context.get('available_budget', 0)
    desktop_rec = context.get('desktop_recommended', 0)
    laptop_rec = context.get('laptop_recommended', 0)
    printer_rec = context.get('printer_recommended', 0)

    items_str = ', '.join(selected_items) if selected_items else 'None selected'
    status = 'WITHIN BUDGET' if required_budget <= available_budget else 'OVER BUDGET'

    system_prompt = (
        "You are a senior ICT procurement advisor for the Philippine government. "
        "Generate a concise procurement recommendation report. "
        "Use a professional tone. Structure your response with exactly 4 sections:\n\n"
        "## Procurement Summary\n(Brief 1-2 sentence summary of the procurement need)\n\n"
        "## Analysis\n(2-3 sentences analyzing the data and justifying quantities)\n\n"
        "## Recommendation\n(1-2 sentences with specific actionable procurement advice)\n\n"
        "## Why\n(1-2 sentences explaining the reasoning and priority logic)"
    )

    user_prompt = (
        f"Selected Items: {items_str}\n"
        f"Total Employees: {emp_count}\n"
        f"Employees Without Computer: {without_pc}\n"
        f"Offices Needing Procurement: {offices_needing}\n"
        f"Recommended Desktop Computers: {desktop_rec}\n"
        f"Recommended Laptop Computers: {laptop_rec}\n"
        f"Recommended Printers: {printer_rec}\n"
        f"Required Budget: PHP {required_budget:,.0f}\n"
        f"Available Budget: PHP {available_budget:,.0f}\n"
        f"Budget Status: {status}"
    )

    result = query_llm(model_name, system_prompt, user_prompt, temperature=0.3)
    return result


def generate_employee_recommendation(model_name, employee_data):
    """Generate personalized employee procurement recommendation with explanation."""
    emp_name = employee_data.get('employee_name', 'Unknown')
    office = employee_data.get('office_division', 'Unknown')
    status = employee_data.get('status_of_employment', 'Unknown')
    nature = employee_data.get('nature_of_work', 'Unknown')
    priority_score = employee_data.get('priority_score', 0)
    recommendation = employee_data.get('recommendation', 'Unknown')

    system_prompt = (
        "You are an ICT asset allocation specialist. "
        "For the given employee, provide a brief personalized recommendation. "
        "Structure your response with exactly these 3 sections:\n\n"
        "## Analysis\n(Brief analysis of the employee's situation)\n\n"
        "## Recommendation\n(Specific equipment recommendation)\n\n"
        "## Why\n(Reasoning behind the recommendation)"
    )

    user_prompt = (
        f"Employee: {emp_name}\n"
        f"Office: {office}\n"
        f"Employment Status: {status}\n"
        f"Nature of Work: {nature}\n"
        f"Priority Score: {priority_score}\n"
        f"System Recommendation: {recommendation}"
    )

    result = query_llm(model_name, system_prompt, user_prompt, temperature=0.3)
    return result
