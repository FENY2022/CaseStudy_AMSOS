"""
AI-Based ICT Asset Replacement and Procurement Decision Support System
Streamlit Interactive Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import re
import ollama
import json
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title='ICT AI-DSS Dashboard',
    page_icon='💻',
    layout='wide',
    initial_sidebar_state='expanded'
)

CURRENT_YEAR = 2026
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'
DESKTOP_COST = 50000
LAPTOP_COST = 55000

def compute_employee_priority(inv_df):
    """
    Determine which unique employees need a computer.
    Uses employeeName to identify unique employees.
    """
    # Deduplicate employees by employeeName
    unique_emps = inv_df.drop_duplicates(subset='employeeName')

    # Identify employees who already own a Desktop or Laptop
    comps = inv_df[
        inv_df['equipmentType'].str.contains('Desktop|Laptop', case=False, na=False)
    ]
    emps_with_computer = comps['employeeName'].dropna().unique()

    # Employees needing a computer = all unique employees minus those with a computer
    needs_computer = unique_emps[~unique_emps['employeeName'].isin(emps_with_computer)].copy()

    employees_no_pc = []
    for _, row in needs_computer.iterrows():
        employees_no_pc.append({
            'employee_name': row['employeeName'],
            'sex': row.get('sex', 'Unknown'),
            'office_division': row.get('officeDivision', 'Unknown'),
            'status_of_employment': row.get('statusOfEmployment', 'Unknown'),
            'nature_of_work': row.get('natureOfWork', 'Unknown'),
        })

    return pd.DataFrame(employees_no_pc)


def compute_employee_priority_score(df):
    if df.empty:
        return df

    def score(row):
        s = 0
        status = str(row['status_of_employment']).lower()
        nature = str(row['nature_of_work']).lower()
        if 'permanent' in status:
            s += 30
        elif 'casual' in status or 'cotractual' in status:
            s += 15
        elif 'contract' in status or 'job' in status or 'cos' in status:
            s += 10
        if 'technical' in nature:
            s += 30
        elif 'administrative' in nature or 'clerical' in nature:
            s += 20
        else:
            s += 10
        return s

    df['priority_score'] = df.apply(score, axis=1)

    def recommendation(row):
        score = row['priority_score']
        nature = str(row['nature_of_work']).lower()
        if score >= 50:
            return 'Assign Desktop Computer'
        elif score >= 30:
            return 'Assign Desktop or Laptop'
        else:
            return 'No Procurement - Shared workstation sufficient'

    df['recommendation'] = df.apply(recommendation, axis=1)
    df = df.sort_values('priority_score', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)
    return df


def compute_procurement(emp_priority):
    """
    Procurement based ONLY on unique employees that need a computer.
    """
    rec_counts = emp_priority['recommendation'].value_counts()

    emp_need_desktop = int(rec_counts.get('Assign Desktop Computer', 0))
    emp_need_either = int(rec_counts.get('Assign Desktop or Laptop', 0))
    emp_no_proc = int(rec_counts.get('No Procurement - Shared workstation sufficient', 0))

    need_computer = emp_priority[
        emp_priority['recommendation'].str.contains('Desktop|Laptop', na=False)
    ]
    tech_mask = need_computer['nature_of_work'].str.contains(
        'Technical', case=False, na=False
    )
    rec_desktop = int(need_computer[tech_mask].shape[0])
    rec_laptop = int(need_computer[~tech_mask].shape[0])

    desktop_budget = rec_desktop * DESKTOP_COST
    laptop_budget = rec_laptop * LAPTOP_COST
    total_budget = desktop_budget + laptop_budget

    proc = pd.DataFrame([{
        'category': 'Desktop Computer',
        'recommended_purchase': rec_desktop,
        'unit_cost': DESKTOP_COST,
        'estimated_budget': desktop_budget,
        'notes': f'Employees needing computer with Technical work: {rec_desktop}'
    }, {
        'category': 'Laptop Computer',
        'recommended_purchase': rec_laptop,
        'unit_cost': LAPTOP_COST,
        'estimated_budget': laptop_budget,
        'notes': f'Employees needing computer with Administrative/other work: {rec_laptop}'
    }, {
        'category': 'TOTAL',
        'recommended_purchase': rec_desktop + rec_laptop,
        'unit_cost': 0,
        'estimated_budget': total_budget,
        'notes': 'Total procurement budget for employees without computer'
    }])

    return proc


def compute_division_shortage(inv_df, div_df):
    """Count unique employees (by employeeName) with computers per division."""
    unique_emps = inv_df[['employeeName', 'officeDivision']].drop_duplicates(subset='employeeName')

    comps = inv_df[inv_df['equipmentType'].str.contains('Desktop|Laptop', case=False, na=False)]
    comps_unique = comps[['employeeName', 'officeDivision']].drop_duplicates(subset='employeeName')
    comps_per_div = comps_unique.groupby('officeDivision').size().reset_index(name='assigned_computers')

    div_df['Count'] = pd.to_numeric(div_df['Count'], errors='coerce').fillna(0).astype(int)
    shortage = div_df.merge(comps_per_div, left_on='Division', right_on='officeDivision', how='left')
    shortage['assigned_computers'] = shortage['assigned_computers'].fillna(0).astype(int)
    shortage['shortage'] = (shortage['Count'] - shortage['assigned_computers']).clip(lower=0)
    shortage['recommended_procurement'] = shortage['shortage']
    shortage = shortage.sort_values('recommended_procurement', ascending=False).reset_index(drop=True)

    result = shortage[['Division', 'Count', 'assigned_computers', 'shortage']]
    result['replacement_count'] = 0
    return result


# === PROCUREMENT POLICY-BASED ANALYSIS FUNCTIONS ===

# Standard unit costs for all procurement items
EQUIPMENT_COSTS = {
    'Desktop Computer': 50000,
    'Laptop Computer': 55000,
    'Printer': 15000,
    'Scanner': 12000,
    'Monitor': 8000,
    'UPS': 5000,
    'Projector': 25000,
    'Router': 3000,
    'Switch': 4000,
    'CCTV': 10000,
}

ALL_PROCUREMENT_ITEMS = [
    'Desktop Computer', 'Laptop Computer', 'Printer', 'Scanner',
    'Monitor', 'UPS', 'Projector', 'Router', 'Switch', 'CCTV'
]

# Shared equipment policies per office
# (item_name, max_allowed, inventory_variants_for_counting)
SHARED_POLICIES = [
    ('Printer', 3, ['Printer']),
    ('CCTV', 4, ['CCTV / IP CAMERA', 'CCTV']),
    ('Projector', 1, ['LCD PROJECTORS', 'Projector']),
    ('Switch', 1, ['Network Switches', 'Switch']),
    ('Router', 1, ['Routers', 'Router']),
]


def normalize_office(name):
    """Normalise office division names to avoid duplicates."""
    if pd.isna(name):
        return name
    name = str(name).strip()
    if name.lower() == 'legal':
        return 'LEGAL'
    return name


def build_employee_equipment_map(inv_df):
    """
    Step 1: Remove duplicate employee names – each employee counts once.
    Step 2: Collect the set of equipment each unique employee owns.
    """
    unique_emps = inv_df[[
        'employeeName', 'officeDivision', 'sex', 'statusOfEmployment', 'natureOfWork'
    ]].drop_duplicates(subset='employeeName').reset_index(drop=True)
    # Normalise office names
    unique_emps['officeDivision'] = unique_emps['officeDivision'].apply(normalize_office)

    emp_equip = inv_df.groupby('employeeName')['equipmentType'].apply(
        lambda x: set(x.dropna())
    ).reset_index()
    emp_equip.columns = ['employeeName', 'equipment_set']

    result = unique_emps.merge(emp_equip, on='employeeName', how='left')
    result['equipment_set'] = result['equipment_set'].apply(
        lambda s: s if isinstance(s, set) else set()
    )
    return result


def compute_office_equipment_counts(inv_df):
    """
    Count total records per officeDivision for each equipment type.
    Used for shared-resource policy decisions (printers, CCTV, etc.).
    """
    df = inv_df.copy()
    df['officeDivision'] = df['officeDivision'].apply(normalize_office)

    offices = df['officeDivision'].unique()
    result = {}
    for office in offices:
        odf = df[df['officeDivision'] == office]
        counts = {}
        for item, _, variants in SHARED_POLICIES:
            pat = '|'.join(variants)
            counts[item] = int(odf[odf['equipmentType'].str.contains(pat, case=False, na=False)].shape[0])
        # Also count Desktop and Laptop per office (unique employees)
        unique_in_office = df[df['officeDivision'] == office]['employeeName'].unique()
        counts['Desktop Computer'] = int(df[(df['officeDivision'] == office) &
            (df['equipmentType'].str.contains('Desktop Computer', case=False, na=False))
            ]['employeeName'].nunique())
        counts['Laptop Computer'] = int(df[(df['officeDivision'] == office) &
            (df['equipmentType'].str.contains('Laptop', case=False, na=False))
            ]['employeeName'].nunique())
        # UPS count (total records, shared per office)
        counts['UPS'] = int(odf[odf['equipmentType'].str.contains('UPS', case=False, na=False)].shape[0])
        result[office] = counts
    return result


def compute_employee_computer_procurement(emp_map, selected_items):
    """
    Policy: Desktop/Laptop only for employees with NO Desktop AND NO Laptop.
    If both selected: Technical work → Desktop, others → Laptop.
    """
    want_desktop = 'Desktop Computer' in selected_items
    want_laptop = 'Laptop Computer' in selected_items

    # Employees with neither Desktop nor Laptop
    no_computer = emp_map[emp_map['equipment_set'].apply(
        lambda s: not any(v in s for v in ['Desktop Computer', 'Laptop Computer', 'Laptop Computers'])
    )].copy()

    results = []
    for _, row in no_computer.iterrows():
        nature = str(row.get('natureOfWork', '')).lower()
        # If both selected, split by work nature
        if want_desktop and (not want_laptop or 'technical' in nature):
            results.append({
                'employeeName': row['employeeName'],
                'office': row['officeDivision'],
                'item': 'Desktop Computer',
                'cost': EQUIPMENT_COSTS['Desktop Computer'],
                'reason': 'Employee has no assigned Desktop or Laptop computer.',
            })
        elif want_laptop:
            results.append({
                'employeeName': row['employeeName'],
                'office': row['officeDivision'],
                'item': 'Laptop Computer',
                'cost': EQUIPMENT_COSTS['Laptop Computer'],
                'reason': 'Employee has no assigned Desktop or Laptop computer.',
            })
    return pd.DataFrame(results)


def compute_shared_procurement(office_counts, selected_items):
    """
    Apply shared-resource procurement policies per office.
    """
    rows = []
    for item, max_allowed, variants in SHARED_POLICIES:
        if item not in selected_items:
            continue
        for office, counts in office_counts.items():
            current = counts.get(item, 0)
            recommend = int(max(0, max_allowed - current))
            if recommend > 0:
                # Build policy explanation
                if item == 'Printer':
                    explanation = (
                        f"The procurement policy allows a maximum of {max_allowed} shared printers per division. "
                        f"Since {office} currently has {current} printer(s), "
                        f"the AI recommends procuring {recommend} additional printer(s)."
                    )
                elif item == 'CCTV':
                    explanation = (
                        f"The procurement policy recommends at least {max_allowed} CCTV units per office for security coverage. "
                        f"Since {office} currently has {current} CCTV unit(s), "
                        f"the AI recommends procuring {recommend} additional unit(s)."
                    )
                else:
                    explanation = (
                        f"The procurement policy allows a maximum of {max_allowed} {item}(s) per division. "
                        f"Since {office} currently has {current}, "
                        f"the AI recommends procuring {recommend} unit(s)."
                    )
                rows.append({
                    'office': office,
                    'item': item,
                    'units': recommend,
                    'cost_per_unit': EQUIPMENT_COSTS.get(item, 0),
                    'reason': explanation,
                })
    return pd.DataFrame(rows)


def compute_ups_procurement(emp_map, desktop_recs, selected_items):
    """
    UPS: recommend for every employee getting a Desktop who does not already own UPS.
    """
    if 'UPS' not in selected_items or desktop_recs.empty:
        return pd.DataFrame()
    results = []
    for _, rec in desktop_recs.iterrows():
        emp_data = emp_map[emp_map['employeeName'] == rec['employeeName']]
        if not emp_data.empty and 'UPS' not in emp_data.iloc[0]['equipment_set']:
            results.append({
                'employeeName': rec['employeeName'],
                'office': rec['office'],
                'item': 'UPS',
                'cost': EQUIPMENT_COSTS['UPS'],
                'reason': f"Employee {rec['employeeName']} is recommended a Desktop Computer but currently has no UPS.",
            })
    return pd.DataFrame(results)


def policy_full_analysis(emp_map, office_counts, selected_items):
    """
    Run all policy-based computations and return:
      emp_recs_df   – per-employee Desktop/Laptop recommendations
      shared_df     – per-office shared-resource recommendations
      ups_df        – per-employee UPS recommendations
      offices_df    – per-office summary
      proc_list_df  – combined procurement list
      summary       – per-item totals
      total_units, total_budget
    """
    # 1) Per-employee Desktop / Laptop
    emp_recs = compute_employee_computer_procurement(emp_map, selected_items)

    # 2) Shared resources per office
    shared_df = compute_shared_procurement(office_counts, selected_items)

    # 3) UPS for Desktop recipients
    desktop_recs = emp_recs[emp_recs['item'] == 'Desktop Computer'] if not emp_recs.empty else pd.DataFrame()
    ups_df = compute_ups_procurement(emp_map, desktop_recs, selected_items)

    # 4) Office summary
    office_records = []
    for office, group in emp_map.groupby('officeDivision'):
        total_emps = len(group)
        w_desk = int(group['equipment_set'].apply(lambda s: 'Desktop Computer' in s).sum())
        w_lap = int(group['equipment_set'].apply(
            lambda s: any(v in s for v in ['Laptop Computer', 'Laptop Computers'])).sum())
        w_comp = int(group['equipment_set'].apply(
            lambda s: any(v in s for v in ['Desktop Computer', 'Laptop Computer', 'Laptop Computers'])).sum())
        wo_comp = total_emps - w_comp

        rec_items = {}
        for item in selected_items:
            if item in ('Desktop Computer', 'Laptop Computer'):
                cnt = int(emp_recs[(emp_recs['item'] == item) & (emp_recs['office'] == office)].shape[0]) if not emp_recs.empty else 0
            elif item == 'UPS':
                cnt = int(ups_df[ups_df['office'] == office].shape[0]) if not ups_df.empty else 0
            else:
                cnt = int(shared_df[(shared_df['item'] == item) & (shared_df['office'] == office)]['units'].sum()) if not shared_df.empty else 0
            rec_items[f'rec_{item}'] = cnt

        budget = sum(rec_items.get(f'rec_{item}', 0) * EQUIPMENT_COSTS.get(item, 0) for item in selected_items)

        # Existing printer / CCTV counts for display
        o_counts = office_counts.get(office, {})
        existing_printers = o_counts.get('Printer', 0)
        existing_cctv = o_counts.get('CCTV', 0)

        office_records.append({
            'Office': office,
            'Total Employees': total_emps,
            'Employees with Desktop': w_desk,
            'Employees with Laptop': w_lap,
            'Employees with Computer': w_comp,
            'Employees without Computer': wo_comp,
            'Existing Printers': existing_printers,
            'Existing CCTV': existing_cctv,
            **rec_items,
            'Estimated Budget': budget,
        })

    offices_df = pd.DataFrame(office_records)
    if not offices_df.empty:
        offices_df = offices_df.sort_values('Employees without Computer', ascending=False).reset_index(drop=True)
        offices_df['Rank'] = range(1, len(offices_df) + 1)
        max_need = offices_df['Employees without Computer'].max()
        if max_need > 0:
            offices_df['Priority'] = offices_df['Employees without Computer'].apply(
                lambda n: 'HIGH' if n >= max_need * 0.5 else ('MEDIUM' if n >= max_need * 0.2 else 'LOW')
            )
        else:
            offices_df['Priority'] = 'LOW'

    # 5) Procurement list (combine employee + shared)
    proc_rows = []
    # Employee-level items
    for _, r in emp_recs.iterrows():
        proc_rows.append({
            'Employee': r['employeeName'],
            'Office': r['office'],
            'Item': r['item'],
            'Type': 'Per-Employee',
            'Quantity': 1,
            'Unit Cost': r['cost'],
            'Total Cost': r['cost'],
            'Reason': r['reason'],
        })
    for _, r in ups_df.iterrows():
        proc_rows.append({
            'Employee': r['employeeName'],
            'Office': r['office'],
            'Item': r['item'],
            'Type': 'Per-Employee',
            'Quantity': 1,
            'Unit Cost': r['cost'],
            'Total Cost': r['cost'],
            'Reason': r['reason'],
        })
    # Shared items (office-level)
    for _, r in shared_df.iterrows():
        total = r['units'] * r['cost_per_unit']
        proc_rows.append({
            'Employee': f'{r["office"]} (Office)',
            'Office': r['office'],
            'Item': r['item'],
            'Type': 'Shared',
            'Quantity': int(r['units']),
            'Unit Cost': int(r['cost_per_unit']),
            'Total Cost': int(total),
            'Reason': r['reason'],
        })

    proc_list = pd.DataFrame(proc_rows)
    if not proc_list.empty:
        proc_list = proc_list.sort_values('Total Cost', ascending=False).reset_index(drop=True)

    # 6) Summary
    summary = {}
    for item in selected_items:
        units = 0
        if not emp_recs.empty:
            units += int(emp_recs[emp_recs['item'] == item].shape[0])
        if item == 'UPS' and not ups_df.empty:
            units += int(ups_df.shape[0])
        if not shared_df.empty:
            units += int(shared_df[shared_df['item'] == item]['units'].sum())
        cost = EQUIPMENT_COSTS.get(item, 0)
        summary[item] = {'units': units, 'unit_cost': cost, 'total': units * cost}

    total_units = sum(v['units'] for v in summary.values())
    total_budget = sum(v['total'] for v in summary.values())

    return emp_recs, shared_df, ups_df, offices_df, proc_list, summary, total_units, total_budget


@st.cache_resource
def load_models():
    clf = joblib.load(MODEL_DIR / 'replacement_model.pkl')
    reg = joblib.load(MODEL_DIR / 'health_score_model.pkl')
    encoder = joblib.load(MODEL_DIR / 'label_encoder.pkl')
    feature_cols = joblib.load(MODEL_DIR / 'feature_columns.pkl')
    return clf, reg, encoder, feature_cols


@st.cache_data
def load_data():
    inv = pd.read_csv(DATA_DIR / 'inv_inventory.csv', low_memory=False)
    repair = pd.read_csv(DATA_DIR / 'repairhistory.csv', low_memory=False)
    div = pd.read_csv(DATA_DIR / 'division_counts.csv', low_memory=False)
    repl = pd.read_csv(OUTPUT_DIR / 'replacement_priority.csv') if (OUTPUT_DIR / 'replacement_priority.csv').exists() else None

    # Compute employee priority using corrected unique-employee logic
    emp = compute_employee_priority(inv)
    if len(emp) > 0:
        emp = compute_employee_priority_score(emp)
    # Compute division shortage using unique employees per division
    div_short = compute_division_shortage(inv, div)
    # Compute procurement based ONLY on employees needing a computer
    proc = compute_procurement(emp) if len(emp) > 0 else None
    return inv, repair, div, repl, emp, div_short, proc

with st.spinner('Loading models and data...'):
    clf, reg, encoder, feature_cols = load_models()
    inv, repair, div, repl_df, emp_df, div_short, proc = load_data()

@st.cache_resource(ttl=300)
def get_ollama_models():
    try:
        result = ollama.list()
        models = []
        for m in result.models:
            family = m.details.family if m.details and m.details.family else ''
            if family not in ('nomic-bert', ''):
                models.append(m.model)
        return models
    except Exception as e:
        return []

ollama_models = get_ollama_models()

st.sidebar.title('🗂️ ICT AI-DSS')
st.sidebar.markdown('---')
page = st.sidebar.radio('Navigation', [
    '📊 Dashboard KPIs',
    '🧠 AI Critical Decisions',
    '🖥️ Replacement Priority',
    '👥 Employee Priority',
    '🏢 Division Shortage',
    '💰 Procurement Budget',
    '📈 Visualizations',
    '🤖 AI Assistant'
])
st.sidebar.markdown('---')
st.sidebar.caption(f'Total Assets: {len(inv)}')
st.sidebar.caption(f'Total Computers: {len(inv[inv["equipmentType"].str.contains("Desktop|Laptop", case=False, na=False)])}')

if page == '📊 Dashboard KPIs':
    st.title('📊 ICT Asset Decision Support Dashboard')
    st.markdown('---')

    kpi_cols = st.columns(5)
    total_computers = len(inv[inv['equipmentType'].str.contains('Desktop|Laptop', case=False, na=False)])
    inv['yearAcquired'] = pd.to_numeric(inv['yearAcquired'], errors='coerce')
    beyond_5yr = len(inv[inv['yearAcquired'] <= CURRENT_YEAR - 6])
    beyond_life = len(inv[inv['shelfLife'] == 'Beyond 5 year'])
    total_repairs = len(repair)

    kpi_cols[0].metric('Total Assets', len(inv))
    kpi_cols[1].metric('Total Computers', total_computers)
    kpi_cols[2].metric('Assets >5 Years', beyond_5yr, delta=f'{beyond_5yr/len(inv)*100:.0f}%')
    kpi_cols[3].metric('Beyond Useful Life', beyond_life)
    kpi_cols[4].metric('Repair Requests', total_repairs)

    kpi_cols2 = st.columns(4)
    if repl_df is not None:
        avg_health = repl_df['asset_health_score'].mean()
        critical = len(repl_df[repl_df['predicted_priority'] == 'Critical'])
        high = len(repl_df[repl_df['predicted_priority'] == 'High'])
        kpi_cols2[0].metric('Avg Asset Health', f'{avg_health:.1f}/100')
        kpi_cols2[1].metric('Critical Assets', critical)
        kpi_cols2[2].metric('High Priority', high)
    if emp_df is not None:
        kpi_cols2[3].metric('Employees w/o PC', len(emp_df))

    st.markdown('---')
    st.subheader('📋 Recent Predictions')
    if repl_df is not None:
        cols = st.columns([1, 1])
        with cols[0]:
            priority_counts = repl_df['predicted_priority'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            colors_p = {'Critical': '#1b5e20', 'High': '#2e7d32', 'Medium': '#388e3c', 'Low': '#66bb6a'}
            bar_colors = [colors_p.get(p, '#a8e6a3') for p in priority_counts.index]
            ax.bar(priority_counts.index, priority_counts.values, color=bar_colors, edgecolor='white')
            ax.set_title('Replacement Priority Distribution', fontweight='bold')
            ax.set_ylabel('Count')
            for i, v in enumerate(priority_counts.values):
                ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
            st.pyplot(fig)
        with cols[1]:
            if 'health_category' in repl_df.columns:
                health_cats = repl_df['health_category'].value_counts()
            else:
                health = repl_df['asset_health_score']
                bins = [0, 20, 40, 60, 80, 90, 100]
                labels = ['Replace', 'Poor', 'Aging', 'Good', 'Very Good', 'Excellent']
                health_cats = pd.cut(health, bins=bins, labels=labels, right=False).value_counts()
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            colors_h = ['#1b5e20', '#2e7d32', '#388e3c', '#43a047', '#66bb6a', '#81c784']
            ax2.bar(health_cats.index, health_cats.values, color=colors_h[:len(health_cats)], edgecolor='white')
            ax2.set_title('Asset Health Distribution', fontweight='bold')
            ax2.set_ylabel('Count')
            for i, v in enumerate(health_cats.values):
                ax2.text(i, v + 2, str(v), ha='center', fontweight='bold')
            st.pyplot(fig2)

elif page == '🧠 AI Critical Decisions':
    st.title('🧠 AI-Powered Critical Decision Engine')
    st.markdown('The AI applies **critical reasoning** to decide replacement priority and generate procurement recommendations with full justification.')

    if not ollama_models:
        st.error('⚠️ No Ollama models available. Ensure Ollama is running.')
    else:
        ai_model = st.sidebar.selectbox('🧠 Decision Model', ollama_models, key='dec_model',
                                        index=ollama_models.index('qwen3:latest') if 'qwen3:latest' in ollama_models else 0)
        st.sidebar.caption(f'Decision model: {ai_model}')

        tab1, tab2, tab3, tab4 = st.tabs([
            '🔍 Decide Per Asset', '📝 AI Replacement Report', '📋 AI Procurement Plan', '📊 AI vs ML Comparison'
        ])

        with tab1:
            st.subheader('AI-Driven Asset Replacement Decision')
            if repl_df is not None:
                cols = st.columns([2, 1])
                with cols[0]:
                    asset_search = st.text_input('🔍 Search asset by property number, user, or equipment', key='ai_search')
                with cols[1]:
                    priority_filter_ai = st.selectbox('Filter by priority',
                                                      ['All'] + list(repl_df['predicted_priority'].unique()), key='ai_prior')

                search_mask = pd.Series(True, index=repl_df.index)
                if asset_search:
                    search_mask = repl_df['propertyNumber'].astype(str).str.contains(asset_search, case=False, na=False) | \
                                  repl_df['actualUser'].astype(str).str.contains(asset_search, case=False, na=False) if 'actualUser' in repl_df.columns else search_mask
                if priority_filter_ai != 'All':
                    search_mask &= repl_df['predicted_priority'] == priority_filter_ai

                candidates = repl_df[search_mask].head(20).copy()

                if len(candidates) > 0:
                    display = candidates[['rank', 'propertyNumber', 'equipmentType', 'brand',
                                          'asset_health_score', 'predicted_priority']].copy()
                    st.dataframe(display, use_container_width=True)

                    selected_idx = st.selectbox('Select asset for AI critical reasoning:',
                                                candidates.index, format_func=lambda x: f"{candidates.loc[x,'propertyNumber']} - {candidates.loc[x,'equipmentType']}")

                    if st.button('🧠 Run AI Critical Reasoning', type='primary', use_container_width=True):
                        asset = candidates.loc[selected_idx]
                        asset_detail = inv[inv['id'] == asset.get('rank', 0)]
                        if len(asset_detail) == 0:
                            asset_detail = inv.iloc[0:1]

                        spec = asset_detail['specifications'].values[0] if 'specifications' in asset_detail.columns else 'N/A'
                        year = asset_detail['yearAcquired'].values[0] if 'yearAcquired' in asset_detail.columns else 'N/A'
                        age = CURRENT_YEAR - int(year) if str(year).isdigit() else 'N/A'
                        emp_status = asset_detail['statusOfEmployment'].values[0] if 'statusOfEmployment' in asset_detail.columns else 'N/A'
                        nature = asset_detail['natureOfWork'].values[0] if 'natureOfWork' in asset_detail.columns else 'N/A'
                        division = asset_detail['officeDivision'].values[0] if 'officeDivision' in asset_detail.columns else 'N/A'

                        prompt = f"""You are an ICT Asset Replacement Decision Expert. Use critical reasoning to analyze the following asset and decide its replacement priority.

ASSET DETAILS:
- Property Number: {asset.get('propertyNumber', 'N/A')}
- Equipment Type: {asset.get('equipmentType', 'N/A')}
- Brand: {asset.get('brand', 'N/A')}
- Year Acquired: {year}
- Equipment Age: {age} years
- Specifications: {spec}
- Accountable Person: {asset.get('accountablePerson', 'N/A')}
- Actual User: {asset.get('actualUser', 'N/A')}
- Office Division: {division}
- Employment Status: {emp_status}
- Nature of Work: {nature}
- ML-Based Health Score: {asset.get('asset_health_score', 'N/A')}/100
- ML Predicted Priority: {asset.get('predicted_priority', 'N/A')}
- ML Prediction Probability: {asset.get('prediction_probability', 'N/A')}
- Total Repairs: {asset.get('total_repairs', 0)}
- Replacement Score: {asset.get('replacement_score', 'N/A')}

CRITICAL REASONING FRAMEWORK:
1. **Age Analysis**: Is the equipment beyond its useful life (5 years)?
2. **Repair History**: How many repairs? Are repairs becoming frequent?
3. **Hardware Viability**: Are the specs (CPU, RAM, Storage) still adequate for the user's work nature?
4. **Depreciation**: Is the asset fully depreciated?
5. **User Criticality**: Is the user's role critical (technical/permanent)?
6. **License & Software**: Are licenses at risk?
7. **Cost-Benefit**: Is it cheaper to replace than to keep repairing?

You MUST output a structured decision in this EXACT format:

**DECISION: [Critical / High / Medium / Low]**

**CONFIDENCE: [0-100]%**

**REASONING:**
- Age Factor: [analysis]
- Repair Factor: [analysis]
- Hardware Factor: [analysis]
- User Factor: [analysis]
- Financial Factor: [analysis]

**RECOMMENDATION:**
[Clear action: Replace Immediately / Plan Replacement / Monitor / No Action]

**ESTIMATED REPLACEMENT COST:**
Desktop: ₱50,000 | Laptop: ₱55,000 | Recommend: [type]"""

                        with st.chat_message('assistant'):
                            with st.spinner('AI analyzing with critical reasoning...'):
                                msg_placeholder = st.empty()
                                full_response = ''
                                try:
                                    stream = ollama.chat(
                                        model=ai_model,
                                        messages=[{'role': 'user', 'content': prompt}],
                                        stream=True
                                    )
                                    for chunk in stream:
                                        if 'message' in chunk and 'content' in chunk['message']:
                                            full_response += chunk['message']['content']
                                            msg_placeholder.markdown(full_response + '▌')
                                    msg_placeholder.markdown(full_response)
                                except Exception as e:
                                    st.error(f'Ollama error: {e}')

                        with st.expander('📋 AI Decision Summary'):
                            st.markdown(full_response)
                else:
                    st.info('No assets match your search.')
            else:
                st.warning('Run Notebook 03 first.')

        with tab2:
            st.subheader('📝 Generate AI-Driven Replacement Report')
            st.markdown('Have the AI analyze ALL assets and generate a comprehensive replacement report with critical reasoning.')
            
            if st.button('🚀 Generate Full AI Replacement Report', type='primary', use_container_width=True):
                with st.spinner('AI generating comprehensive replacement report...'):
                    if repl_df is not None:
                        critical_count = len(repl_df[repl_df['predicted_priority'] == 'Critical'])
                        high_count = len(repl_df[repl_df['predicted_priority'] == 'High'])
                        medium_count = len(repl_df[repl_df['predicted_priority'] == 'Medium'])
                        low_count = len(repl_df[repl_df['predicted_priority'] == 'Low'])
                        avg_health = repl_df['asset_health_score'].mean()
                        total_budget = critical_count * 50000 + high_count * 50000

                        prompt = f"""You are a Senior ICT Asset Management Director. Generate a comprehensive AI-Driven Replacement Report.

DATA SUMMARY:
- Total Assets: {len(inv)}
- Critical Priority: {critical_count}
- High Priority: {high_count}
- Medium Priority: {medium_count}
- Low Priority: {low_count}
- Average Health Score: {avg_health:.1f}/100
- Total Estimated Budget Needed: ₱{total_budget:,}

STRUCTURE YOUR REPORT:
1. **Executive Summary** - Overall assessment
2. **Critical Assets Analysis** - Why they need immediate replacement
3. **High Priority Assets** - Which need planned replacement
4. **Division-Level Impact** - Which divisions are most affected
5. **Budget Recommendation** - Justified budget with phasing
6. **Timeline** - Urgent (<3mo), Short-term (3-6mo), Medium (6-12mo)
7. **Risk Assessment** - What happens if replacement is delayed

Be specific, use critical reasoning, and provide actionable recommendations."""

                        with st.chat_message('assistant'):
                            msg = st.empty()
                            full = ''
                            try:
                                stream = ollama.chat(model=ai_model, messages=[{'role': 'user', 'content': prompt}], stream=True)
                                for chunk in stream:
                                    if 'message' in chunk and 'content' in chunk['message']:
                                        full += chunk['message']['content']
                                        msg.markdown(full + '▌')
                                msg.markdown(full)
                            except Exception as e:
                                st.error(f'Ollama error: {e}')
                    else:
                        st.warning('No data available.')

        with tab3:
            st.subheader('📋 AI-Generated Procurement Plan')
            st.markdown('The AI uses critical reasoning to create a strategic procurement plan.')
            
            if st.button('💰 Generate AI Procurement Plan', type='primary', use_container_width=True):
                with st.spinner('AI analyzing data and generating strategic procurement plan...'):
                    if emp_df is not None and div_short is not None and repl_df is not None:
                        emp_no_pc = len(emp_df)

                        prompt = f"""You are a Government ICT Procurement Strategist. Generate a critical-reasoning-based Procurement Plan.

CURRENT SITUATION:
- Unique Employees Without Computer: {emp_no_pc}
- Desktop Unit Cost: ₱50,000
- Laptop Unit Cost: ₱55,000

REQUIREMENTS:
1. Calculate total procurement needed based on unique employees without computers
2. Recommend Desktop vs Laptop split with justification
3. Provide phased implementation plan (Priority 1, 2, 3)
4. Justify each recommendation with critical reasoning
5. Include risk analysis if procurement is delayed
6. Provide total budget with breakdown

OUTPUT FORMAT:
## PROCUREMENT PLAN
### Phase 1: Immediate (0-3 months)
### Phase 2: Short-term (3-6 months)
### Phase 3: Long-term (6-12 months)
### Total Budget Summary"""

                        with st.chat_message('assistant'):
                            msg = st.empty()
                            full = ''
                            try:
                                stream = ollama.chat(model=ai_model, messages=[{'role': 'user', 'content': prompt}], stream=True)
                                for chunk in stream:
                                    if 'message' in chunk and 'content' in chunk['message']:
                                        full += chunk['message']['content']
                                        msg.markdown(full + '▌')
                                msg.markdown(full)
                            except Exception as e:
                                st.error(f'Ollama error: {e}')

        with tab4:
            st.subheader('📊 AI vs ML Decision Comparison')
            st.markdown('Compare ML model predictions with AI critical-reasoning decisions side-by-side.')
            if repl_df is not None:
                top_assets = repl_df.head(10)
                st.dataframe(top_assets[['rank', 'propertyNumber', 'equipmentType', 'brand',
                                         'predicted_priority', 'replacement_score', 'asset_health_score']],
                             use_container_width=True)

                if st.button('🤖 Compare AI vs ML on these 10 assets', use_container_width=True):
                    progress = st.progress(0, text='Querying AI for each asset...')
                    for i, (idx, asset) in enumerate(top_assets.iterrows()):
                        progress.progress((i + 1) / len(top_assets), text=f'Analyzing asset {i + 1} of {len(top_assets)}...')
                        detail = inv[inv['id'] == asset.get('rank', 0)]
                        spec = detail['specifications'].values[0] if len(detail) > 0 and 'specifications' in detail.columns else 'N/A'
                        year = detail['yearAcquired'].values[0] if len(detail) > 0 and 'yearAcquired' in detail.columns else 'N/A'

                        prompt = f"""Analyze this asset and decide replacement priority.

Property: {asset.get('propertyNumber', 'N/A')}
Type: {asset.get('equipmentType', 'N/A')}
Age: {asset.get('equipment_age', 'N/A')} years
Health Score: {asset.get('asset_health_score', 'N/A')}/100
ML Priority: {asset.get('predicted_priority', 'N/A')}
Repairs: {asset.get('total_repairs', 0)}

Output ONLY one word: Critical, High, Medium, or Low"""

                        with st.spinner('AI predicting replacement priority...'):
                            try:
                                result = ollama.chat(model=ai_model, messages=[{'role': 'user', 'content': prompt}])
                                ai_decision = result['message']['content'].strip().split('\n')[0][:20]
                            except:
                                ai_decision = 'Error'

                        col_a, col_b, col_c = st.columns([2, 1, 1])
                        with col_a:
                            st.caption(f'{asset.get("propertyNumber", "N/A")} - {asset.get("equipmentType", "N/A")}')
                        with col_b:
                            ml_color = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}
                            st.markdown(f'**ML:** {ml_color.get(asset.get("predicted_priority", ""), "")} {asset.get("predicted_priority", "N/A")}')
                        with col_c:
                            ai_color = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}
                            st.markdown(f'**AI:** {ai_color.get(ai_decision, "⚪")} {ai_decision}')

                        match = '✅' if ai_decision.lower() == asset.get('predicted_priority', '').lower() else '⚠️'
                        st.caption(f'{match} ML: {asset.get("predicted_priority", "N/A")} | AI: {ai_decision}')
                        st.divider()

elif page == '🖥️ Replacement Priority':
    st.title('🖥️ Replacement Priority Ranking')
    st.markdown('---')
    if repl_df is not None:
        col_filters = st.columns(4)
        with col_filters[0]:
            priority_filter = st.multiselect('Priority', options=repl_df['predicted_priority'].unique(),
                                              default=list(repl_df['predicted_priority'].unique()))
        with col_filters[1]:
            if 'equipmentType' in repl_df.columns:
                types = st.multiselect('Equipment Type', options=repl_df['equipmentType'].unique(),
                                        default=[])
            else:
                types = []
        with col_filters[2]:
            search = st.text_input('🔍 Search by property # or user')
        with col_filters[3]:
            top_n = st.selectbox('Show top', [20, 50, 100, 200, 500, 1000])

        filtered = repl_df[repl_df['predicted_priority'].isin(priority_filter)]
        if types:
            filtered = filtered[filtered['equipmentType'].isin(types)]
        if search:
            mask = filtered['propertyNumber'].astype(str).str.contains(search, case=False, na=False) | \
                   filtered['actualUser'].astype(str).str.contains(search, case=False, na=False) if 'actualUser' in filtered.columns else False
            filtered = filtered[mask]
        filtered = filtered.head(top_n)

        def color_priority(val):
            colors = {'Critical': 'background-color: #a5d6a7', 'High': 'background-color: #c8e6c9',
                      'Medium': 'background-color: #e8f5e9', 'Low': 'background-color: #f1f8e9'}
            return colors.get(val, '')

        display_cols = [c for c in ['rank', 'propertyNumber', 'equipmentType', 'brand', 'equipment_age',
                                      'total_repairs', 'asset_health_score', 'predicted_priority',
                                      'prediction_probability', 'replacement_score', 'recommendation']
                        if c in filtered.columns]
        st.dataframe(
            filtered[display_cols].style.map(color_priority, subset=['predicted_priority']),
            use_container_width=True,
            height=600
        )
    else:
        st.warning('Run Notebook 03 first to generate replacement priorities.')

elif page == '👥 Employee Priority':
    st.title('👥 Employee Procurement Priority')
    st.markdown('---')
    if emp_df is not None and len(emp_df) > 0:
        st.metric('Employees Without Assigned Computer', len(emp_df))
        st.dataframe(emp_df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            status_counts = emp_df['status_of_employment'].value_counts()
            ax.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
                   startangle=90, colors=plt.cm.Greens(np.linspace(0.3, 0.9, len(status_counts))))
            ax.set_title('Employment Status Distribution', fontweight='bold')
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            nature_counts = emp_df['nature_of_work'].value_counts().head(8)
            ax.barh(nature_counts.index, nature_counts.values, color='#2d5a27', edgecolor='white')
            ax.set_title('Work Nature of Employees w/o PC', fontweight='bold')
            ax.set_xlabel('Count')
            st.pyplot(fig)

        rec_counts = emp_df['recommendation'].value_counts()
        st.subheader('Procurement Recommendations')
        st.dataframe(rec_counts.reset_index(), use_container_width=True)
    else:
        st.info('All employees appear to have assigned computers.')

elif page == '🏢 Division Shortage':
    st.title('🏢 Division Computer Shortage Analysis')
    st.markdown('---')
    if div_short is not None:
        st.metric('Total Computer Shortage', int(div_short['shortage'].sum()))
        st.dataframe(div_short, use_container_width=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        top = div_short.sort_values('shortage', ascending=True).tail(10)
        ax.barh(top['Division'], top['shortage'], color='#3a7d3e', edgecolor='white')
        ax.set_title('Division Computer Shortage', fontweight='bold')
        ax.set_xlabel('Shortage')
        for i, v in enumerate(top['shortage']):
            ax.text(v + 0.3, i, str(v), va='center', fontweight='bold')
        st.pyplot(fig)
    else:
        st.warning('Division shortage data not found.')

elif page == '💰 Procurement Budget':
    st.title('💰 Explainable AI Procurement Decision Support')
    st.markdown('---')

    # ── Sidebar: Procurement Preference ──
    st.sidebar.markdown('---')
    st.sidebar.subheader('🛒 Procurement Preference')
    selected_items = st.sidebar.multiselect(
        'Select items to procure',
        ALL_PROCUREMENT_ITEMS,
        default=['Desktop Computer', 'Laptop Computer'],
        key='proc_items'
    )
    st.sidebar.caption(f'{len(selected_items)} item(s) selected')

    if ollama_models:
        budget_ai_model = st.sidebar.selectbox(
            '🧠 Budget AI Model', ollama_models,
            index=ollama_models.index('qwen3:latest') if 'qwen3:latest' in ollama_models else 0,
            key='budget_ai_model'
        )
        st.sidebar.caption(f'Model: {budget_ai_model}')

    if not selected_items:
        st.info('👈 Please select at least one procurement item from the sidebar.')
    else:
        # ── Build employee equipment map (unique employees only) ──
        emp_map = build_employee_equipment_map(inv)

        total_unique_emps = len(emp_map)
        total_inv_records = len(inv)
        duplicate_records = total_inv_records - total_unique_emps

        # ── Compute all analysis from selected preferences ──
        offices_df = analyze_office_procurement(emp_map, selected_items)
        procurement_list = generate_procurement_list(emp_map, selected_items)
        summary, total_units, total_budget = compute_procurement_summary(procurement_list, selected_items)

        # ── Top-level metric ──
        st.metric(
            'Total Estimated Procurement Budget',
            f'₱{total_budget:,}',
            delta=f'{total_units} unit{"s" if total_units != 1 else ""}'
        )

        # ═══════════════════════════════════════════════════════
        # SECTION 1: AI PROCUREMENT ANALYSIS
        # ═══════════════════════════════════════════════════════
        st.markdown('---')
        st.subheader('🤖 AI Procurement Analysis')

        # Compute computer-specific stats for the narrative
        emps_with_desktop = emp_map['equipment_set'].apply(
            lambda s: 'Desktop Computer' in s
        ).sum()
        emps_with_laptop = emp_map['equipment_set'].apply(
            lambda s: any(v in s for v in ['Laptop Computer', 'Laptop Computers'])
        ).sum()
        emps_with_computer = emp_map['equipment_set'].apply(
            lambda s: any(v in s for v in ['Desktop Computer', 'Laptop Computer', 'Laptop Computers'])
        ).sum()
        emps_without_computer = total_unique_emps - emps_with_computer

        with st.container(border=True):
            # ── Metrics table ──
            cols = st.columns(4)
            cols[0].metric('Total Unique Employees', f'{total_unique_emps:,}')
            cols[1].metric('Duplicate Records Removed', f'{duplicate_records:,}')
            cols[2].metric('Employees with Computer', f'{emps_with_computer:,}')
            cols[3].metric('Employees without Computer', f'{emps_without_computer:,}')

            cols2 = st.columns(3)
            cols2[0].metric('Employees with Desktop', f'{int(emps_with_desktop):,}')
            cols2[1].metric('Employees with Laptop', f'{int(emps_with_laptop):,}')
            if procurement_list is not None and not procurement_list.empty:
                cols2[2].metric('Employees Needing Procurement', f'{len(procurement_list):,}')

            # ── Natural language explanation ──
            st.markdown('**📝 Reason for Recommendation**')

            selected_names = ', '.join(selected_items)
            reason_parts = [
                f"The system analyzed **{total_unique_emps:,}** unique employees across all offices."
            ]
            if duplicate_records > 0:
                reason_parts.append(
                    "Duplicate employee inventory records were automatically consolidated before computation."
                )
            reason_parts.append(
                f"After validating all Desktop and Laptop assignments:"
            )
            reason_parts.append(
                f"• **{int(emps_with_desktop):,}** employees already have Desktop Computers."
            )
            reason_parts.append(
                f"• **{int(emps_with_laptop):,}** employees already have Laptop Computers."
            )
            reason_parts.append(
                f"• **{emps_with_computer:,}** employees already have at least one computer."
            )
            reason_parts.append(
                f"• **{emps_without_computer:,}** employees {'still require' if emps_without_computer != 1 else 'still requires'} ICT equipment."
            )

            if selected_items:
                items_desc = ', '.join(selected_items)
                reason_parts.append(
                    f"Based on the selected procurement preference ({items_desc}), "
                    f"the AI recommends purchasing:"
                )
                for item in selected_items:
                    cnt = summary.get(item, {}).get('units', 0)
                    reason_parts.append(f"• **{item}s**: {cnt:,}")

            reason_text = '\n\n'.join(reason_parts)
            st.markdown(reason_text)

            # ── Duplicate notice ──
            if duplicate_records > 0:
                st.info(
                    f"📌 **Duplicate employee inventory records were detected.** "
                    f"The system consolidated **{total_inv_records:,}** total records into "
                    f"**{total_unique_emps:,}** unique employees to avoid overestimating procurement requirements."
                )

        # ═══════════════════════════════════════════════════════
        # SECTION 2: COMPLETE COMPUTATION
        # ═══════════════════════════════════════════════════════
        st.markdown('---')
        st.subheader('💰 Complete Budget Computation')

        comp_cols = st.columns([1, 1])
        with comp_cols[0]:
            st.markdown('**📊 Total Unique Employees**')
            st.markdown(f'**{total_unique_emps:,}**')

            st.markdown('**📊 Duplicate Records Removed**')
            st.markdown(f'**{duplicate_records:,}**')

            st.markdown('**📊 Employees Already Assigned Desktop**')
            st.markdown(f'**{int(emps_with_desktop):,}**')

            st.markdown('**📊 Employees Already Assigned Laptop**')
            st.markdown(f'**{int(emps_with_laptop):,}**')

            st.markdown('**📊 Employees Already Assigned Computer**')
            st.markdown(f'**{emps_with_computer:,}**')

            st.markdown('**📊 Employees Without Computer**')
            st.markdown(f'**{emps_without_computer:,}**')

        with comp_cols[1]:
            for item in selected_items:
                info = summary.get(item, {'units': 0, 'unit_cost': 0, 'total': 0})
                if info['units'] > 0:
                    st.markdown(f'**📊 Recommended {item}**')
                    st.markdown(f'**{info["units"]:,}** units')

            st.markdown('**📊 Estimated Budget Breakdown**')
            for item in selected_items:
                info = summary.get(item, {'units': 0, 'unit_cost': 0, 'total': 0})
                if info['units'] > 0:
                    st.markdown(
                        f'• **{item}**: {info["units"]:,} × ₱{info["unit_cost"]:,} = '
                        f'₱{info["total"]:,}'
                    )
            st.markdown(f'**Grand Total: ₱{total_budget:,}**')

        # ═══════════════════════════════════════════════════════
        # SECTION 3: OFFICE NEEDS SUMMARY
        # ═══════════════════════════════════════════════════════
        st.markdown('---')
        st.subheader('🏢 Office Needs Summary')
        if offices_df is not None and not offices_df.empty:
            display_cols = ['Office', 'Total Employees', 'Employees with Computer',
                            'Employees without Computer', 'Estimated Budget']
            # Add per-item rec columns
            for item in selected_items:
                col = f'rec_{item}'
                if col in offices_df.columns:
                    display_cols.append(col)
            display_cols.append('Priority')
            available = [c for c in display_cols if c in offices_df.columns]
            st.dataframe(offices_df[available], use_container_width=True)
        else:
            st.info('No office data available.')

        # ═══════════════════════════════════════════════════════
        # SECTION 4: PROCUREMENT PRIORITY RANKING
        # ═══════════════════════════════════════════════════════
        st.markdown('---')
        st.subheader('📊 Procurement Priority Ranking')
        if offices_df is not None and not offices_df.empty:
            rank_df = offices_df[['Rank', 'Office', 'Employees without Computer',
                                  'Estimated Budget', 'Priority']].copy()
            rank_df.columns = ['Rank', 'Office', 'Need', 'Budget', 'Priority']

            def color_priority(val):
                colors = {'HIGH': 'background-color: #1b5e20; color: white',
                          'MEDIUM': 'background-color: #f9a825',
                          'LOW': 'background-color: #e0e0e0'}
                return colors.get(val, '')

            st.dataframe(
                rank_df.style.map(color_priority, subset=['Priority']),
                use_container_width=True
            )
        else:
            st.info('No priority data available.')

        # ═══════════════════════════════════════════════════════
        # SECTION 5: PROCUREMENT RECOMMENDATION LIST
        # ═══════════════════════════════════════════════════════
        st.markdown('---')
        st.subheader('📋 Procurement Recommendation List')
        if procurement_list is not None and not procurement_list.empty:
            st.dataframe(procurement_list, use_container_width=True, height=400)

            # Count by reason
            st.markdown('**Reason Summary**')
            reason_counts = procurement_list['Reason'].value_counts().reset_index()
            reason_counts.columns = ['Reason', 'Employee Count']
            st.dataframe(reason_counts, use_container_width=True)
        else:
            st.info('All employees already have the selected equipment. No procurement needed.')

        # ═══════════════════════════════════════════════════════
        # SECTION 6: BUDGET BREAKDOWN & HOW WAS THIS CALCULATED?
        # ═══════════════════════════════════════════════════════
        st.markdown('---')
        st.subheader('📈 Budget Breakdown')

        if len(selected_items) > 0:
            # Bar chart
            items_chart = [item for item in selected_items if summary.get(item, {}).get('units', 0) > 0]
            if items_chart:
                fig, ax = plt.subplots(figsize=(10, 5))
                categories = items_chart
                values = [summary[item]['total'] for item in items_chart]
                colors = plt.cm.Set2(range(len(categories)))
                bars = ax.bar(categories, values, color=colors, edgecolor='white', width=0.5)
                ax.set_title('Estimated Budget Breakdown by Equipment Type', fontweight='bold')
                ax.set_ylabel('Budget (₱)')
                ax.tick_params(axis='x', rotation=30)
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                            f'₱{val:,}', ha='center', fontweight='bold', fontsize=9)
                st.pyplot(fig)

        # ── Computation steps ──
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('**📐 Mathematical Computation**')
            for item in selected_items:
                info = summary.get(item, {'units': 0, 'unit_cost': 0, 'total': 0})
                if info['units'] > 0:
                    st.markdown(f"""
**{item}**
```
{info['units']:,} × ₱{info['unit_cost']:,}
= ₱{info['total']:,}
```
""")
            st.markdown(f"""
**Grand Total**
```
{' + '.join([f'₱{summary[item]["total"]:,}' for item in selected_items if summary.get(item, {}).get("units", 0) > 0])}
= ₱{total_budget:,}
```
""")

        with col_b:
            with st.expander('**❓ How was this calculated?**', expanded=True):
                st.markdown('**Formula Used**')
                st.markdown("""
```
Unique Employees
        ↓
Employees with Desktop / Laptop
        ↓
Employees without Computer
        ↓
Recommended Purchase
        ↓
Recommended Purchase × Unit Cost
        ↓
Estimated Budget
```
""")
                st.markdown('**AI Confidence**')
                st.markdown('✅ **High**')
                st.caption(
                    'The recommendation is based on unique employee records, existing ICT inventory '
                    'assignments, and standardized procurement costs.'
                )

        # ═══════════════════════════════════════════════════════
        # SECTION 7: AI BUDGET ANALYSIS (Ollama)
        # ═══════════════════════════════════════════════════════
        st.markdown('---')
        st.subheader('🤖 AI Budget Analysis')
        if ollama_models:
            if st.button('📊 Generate AI Budget Analysis', type='primary', use_container_width=True):
                # Build context for the AI
                context_lines = [
                    f"BUDGET DATA:",
                    f"- Total Estimated Budget: ₱{total_budget:,}",
                ]
                for item in selected_items:
                    info = summary.get(item, {'units': 0, 'unit_cost': 0})
                    if info['units'] > 0:
                        context_lines.append(
                            f"- {item}: {info['units']} units @ ₱{info['unit_cost']:,}"
                            f" = ₱{info['units'] * info['unit_cost']:,}"
                        )
                context_lines.append(f"")
                context_lines.append(f"ADDITIONAL CONTEXT:")
                context_lines.append(f"- Unique Employees Without Computer: {emps_without_computer}")
                context_lines.append(f"- Total Unique Employees: {total_unique_emps}")
                context_lines.append(f"- Offices Analyzed: {len(offices_df) if offices_df is not None else 0}")

                prompt = f"""You are a Senior ICT Procurement and Budget Analyst. Analyze the following procurement budget data and provide strategic insights.

{' '.join(context_lines)}

Provide a structured analysis covering:
1. **Budget Adequacy** - Is the budget sufficient to address all needs?
2. **Spending Breakdown** - Is the equipment split appropriate?
3. **Gap Analysis** - What is still needed beyond this budget?
4. **Risk Assessment** - What are the risks of under-funding?
5. **Phasing Recommendation** - How to prioritize spending across phases
6. **Strategic Recommendations** - Actionable next steps"""

                with st.chat_message('assistant'):
                    with st.spinner('AI analyzing budget data with critical reasoning...'):
                        msg_placeholder = st.empty()
                        full_response = ''
                        try:
                            stream = ollama.chat(
                                model=budget_ai_model,
                                messages=[{'role': 'user', 'content': prompt}],
                                stream=True
                            )
                            for chunk in stream:
                                if 'message' in chunk and 'content' in chunk['message']:
                                    full_response += chunk['message']['content']
                                    msg_placeholder.markdown(full_response + '▌')
                            msg_placeholder.markdown(full_response)
                        except Exception as e:
                            st.error(f'Ollama error: {e}')
                            full_response = '⚠️ Error connecting to Ollama. Ensure the server is running.'
                            msg_placeholder.markdown(full_response)

                with st.expander('📋 Full AI Budget Analysis'):
                    st.markdown(full_response)
        else:
            st.info('💡 Start Ollama to unlock AI-powered budget analysis with your preferred model.')

elif page == '📈 Visualizations':
    st.title('📈 System Visualizations')
    st.markdown('---')
    viz_dir = OUTPUT_DIR
    viz_files = list(viz_dir.glob('*.png'))
    viz_names = [f.stem.replace('_', ' ').title() for f in viz_files]

    if viz_files:
        selected = st.selectbox('Select Visualization', viz_names)
        idx = viz_names.index(selected)
        st.image(str(viz_files[idx]), use_column_width=True)
    else:
        st.warning('No visualization files found.')

elif page == '🤖 AI Assistant':
    st.title('🤖 AI-Powered Asset Management Assistant')
    st.markdown('Ask natural language questions about your ICT assets, get AI-powered insights and recommendations.')

    if not ollama_models:
        st.error('⚠️ No Ollama models available. Ensure Ollama is running (`ollama serve`).')
    else:
        with st.sidebar:
            st.markdown('---')
            st.subheader('🤖 AI Settings')
            selected_model = st.selectbox('Ollama Model', ollama_models,
                                          index=ollama_models.index('qwen3:latest') if 'qwen3:latest' in ollama_models else 0)
            system_prompt_type = st.radio('AI Role', [
                'Asset Analyst',
                'Procurement Advisor',
                'Maintenance Expert',
                'Custom'
            ])
            if system_prompt_type == 'Custom':
                custom_prompt = st.text_area('System Prompt', height=100)
            st.caption(f'Model: {selected_model}')

        SYSTEM_PROMPTS = {
            'Asset Analyst': f"""You are an ICT Asset Management Analyst. You have access to the following data:

INVENTORY: {len(inv)} ICT assets including desktops, laptops, printers, etc.
- Total Computers: {len(inv[inv['equipmentType'].str.contains('Desktop|Laptop', case=False, na=False)])}
- Assets Beyond 5 Years: {len(inv[pd.to_numeric(inv['yearAcquired'], errors='coerce') <= CURRENT_YEAR - 6])}
- Repair Records: {len(repair)}

ML MODEL PREDICTIONS:
- Critical Priority: {len(repl_df[repl_df['predicted_priority']=='Critical']) if repl_df is not None else 'N/A'}
- High Priority: {len(repl_df[repl_df['predicted_priority']=='High']) if repl_df is not None else 'N/A'}
- Medium Priority: {len(repl_df[repl_df['predicted_priority']=='Medium']) if repl_df is not None else 'N/A'}
- Low Priority: {len(repl_df[repl_df['predicted_priority']=='Low']) if repl_df is not None else 'N/A'}
- Avg Health Score: {repl_df['asset_health_score'].mean():.1f}/100 if repl_df is not None else 'N/A'

EMPLOYEES WITHOUT COMPUTER: {len(emp_df) if emp_df is not None else 0}
DIVISION SHORTAGE: {int(div_short['shortage'].sum()) if div_short is not None else 0}

Answer questions about asset replacement planning, procurement needs, division shortages, and maintenance priorities. Be specific and reference data when possible.""",

            'Procurement Advisor': f"""You are a Government ICT Procurement Advisor. Analyze the following data to provide procurement recommendations:

Unique Employees without computer: {len(emp_df) if emp_df is not None else 0}

Unit costs: Desktop = ₱{DESKTOP_COST:,}, Laptop = ₱{LAPTOP_COST:,}

Provide budget estimates, procurement priorities, and justification for each recommendation. Base recommendations on unique employees that need a computer, not on inventory record counts.""",

            'Maintenance Expert': f"""You are an ICT Maintenance and Repair Expert. You have:
- {len(repair)} total repair records
- Average asset age: {pd.to_numeric(inv['yearAcquired'], errors='coerce').mean():.0f} years acquired
- {len(inv[inv['shelfLife']=='Beyond 5 year'])} assets beyond useful life

Advise on maintenance scheduling, when to repair vs replace, and how to extend asset life."""
        }

        if system_prompt_type != 'Custom':
            system_prompt = SYSTEM_PROMPTS[system_prompt_type]
        else:
            system_prompt = custom_prompt

        if 'messages' not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({
                'role': 'assistant',
                'content': f'Hello! I am your ICT Asset AI Assistant using **{selected_model}**. Ask me about asset replacement, procurement, division shortages, or maintenance planning.'
            })

        for msg in st.session_state.messages:
            with st.chat_message(msg['role']):
                st.markdown(msg['content'])

        if prompt := st.chat_input('Ask about your ICT assets...'):
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            with st.chat_message('user'):
                st.markdown(prompt)

            with st.chat_message('assistant'):
                with st.spinner(f'AI assistant is thinking with **{selected_model}**...'):
                    msg_placeholder = st.empty()
                    full_response = ''
                    try:
                        stream = ollama.chat(
                            model=selected_model,
                            messages=[{'role': 'system', 'content': system_prompt}] + [
                                {'role': m['role'], 'content': m['content']}
                                for m in st.session_state.messages[-10:]
                            ],
                            stream=True
                        )
                        for chunk in stream:
                            if 'message' in chunk and 'content' in chunk['message']:
                                full_response += chunk['message']['content']
                                msg_placeholder.markdown(full_response + '▌')
                        msg_placeholder.markdown(full_response)
                    except Exception as e:
                        st.error(f'Ollama error: {e}')
                        full_response = '⚠️ Error connecting to Ollama. Ensure the server is running with `ollama serve`.'
                        msg_placeholder.markdown(full_response)

            st.session_state.messages.append({'role': 'assistant', 'content': full_response})
