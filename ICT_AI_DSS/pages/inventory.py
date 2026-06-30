"""ICT Inventory page — full table with search, filter, sort, pagination, export."""

import streamlit as st
import pandas as pd
from ui_components import render_section_header, render_table, to_csv, to_excel


def render(inv):
    render_section_header('📦 ICT Inventory')

    col_filters = st.columns(4)
    with col_filters[0]:
        equip_filter = st.multiselect('Equipment Type', options=sorted(inv['equipmentType'].unique()), default=[], key='inv_equip')
    with col_filters[1]:
        office_filter = st.multiselect('Office/Division', options=sorted(inv['officeDivision'].unique()), default=[], key='inv_office')
    with col_filters[2]:
        status_filter = st.multiselect('Employment Status', options=sorted(inv['statusOfEmployment'].dropna().unique()), default=[], key='inv_status')
    with col_filters[3]:
        search = st.text_input('🔍 Search', placeholder='Property #, name, or equipment...', key='inv_search')

    filtered = inv.copy()
    if equip_filter:
        filtered = filtered[filtered['equipmentType'].isin(equip_filter)]
    if office_filter:
        filtered = filtered[filtered['officeDivision'].isin(office_filter)]
    if status_filter:
        filtered = filtered[filtered['statusOfEmployment'].isin(status_filter)]
    if search:
        mask = filtered.astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)
        filtered = filtered[mask]

    display_cols = ['propertyNumber', 'equipmentType', 'brand', 'employeeName', 'officeDivision', 'statusOfEmployment', 'natureOfWork', 'yearAcquired', 'shelfLife', 'amount']
    available = [c for c in display_cols if c in filtered.columns]

    st.caption(f'{len(filtered)} of {len(inv)} assets')

    page_size = st.selectbox('Show', [10, 25, 50, 100], key='inv_page', index=1)
    total = len(filtered)
    if page_size < total:
        page = st.number_input('Page', 1, max(1, (total + page_size - 1) // page_size), 1, key='inv_pagenum')
        start = (page - 1) * page_size
        end = min(start + page_size, total)
        display = filtered.iloc[start:end]
        st.caption(f'Showing {start+1}–{end} of {total}')
    else:
        display = filtered

    st.dataframe(display[available], use_container_width=True, height=450, hide_index=True)

    export_col1, export_col2 = st.columns(2)
    with export_col1:
        st.download_button('📥 Download CSV', to_csv(filtered[available]), 'ict_inventory.csv', 'text/csv', use_container_width=True)
    with export_col2:
        st.download_button('📥 Download Excel', to_excel(filtered[available]), 'ict_inventory.xlsx', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
