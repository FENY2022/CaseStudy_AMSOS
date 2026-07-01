"""Settings page."""

import streamlit as st
from ui_components import render_section_header


def render(inv):
    render_section_header('⚙ Settings')

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('**System Configuration**')

        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:8px; padding:1rem; margin-bottom:1rem;">
            <div style="font-size:0.8rem; color:#6B7280;">Data Source</div>
            <div style="font-weight:600;">ICT Asset Inventory Database</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:8px; padding:1rem; margin-bottom:1rem;">
            <div style="font-size:0.8rem; color:#6B7280;">Total Records</div>
            <div style="font-weight:600;">{len(inv):,} assets</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:8px; padding:1rem; margin-bottom:1rem;">
            <div style="font-size:0.8rem; color:#6B7280;">Fiscal Year</div>
            <div style="font-weight:600;">2026</div>
        </div>
        """, unsafe_allow_html=True)

        cost_data = [
            ('Desktop Computer', '₱50,000'),
            ('Laptop Computer', '₱55,000'),
            ('Printer', '₱15,000'),
            ('Scanner', '₱12,000'),
            ('Monitor', '₱8,000'),
            ('UPS', '₱5,000'),
            ('Projector', '₱25,000'),
            ('Router', '₱3,000'),
            ('Switch', '₱4,000'),
            ('CCTV', '₱10,000'),
        ]

        st.markdown('**Standard Unit Costs**')
        cost_html = '<div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:8px; padding:1rem;">'
        for item, cost in cost_data:
            cost_html += f'<div style="display:flex; justify-content:space-between; padding:0.25rem 0;"><span>{item}</span><span style="font-weight:600;">{cost}</span></div>'
        cost_html += '</div>'
        st.markdown(cost_html, unsafe_allow_html=True)

    with col2:
        st.markdown('**Printer Allocation Policy**')
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:8px; padding:1rem; margin-bottom:1rem;">
            <p>Shared resource policy:</p>
            <ul>
                <li>Largest division receives max <strong>7 printers</strong></li>
                <li>Other divisions scaled proportionally</li>
                <li><code>Target = ROUND((Div Employees / Largest Div) × 7)</code></li>
                <li><code>Final = MAX(0, Target − Existing)</code></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('**Priority Scoring**')
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:8px; padding:1rem; margin-bottom:1rem;">
            <table style="width:100%; font-size:0.85rem;">
                <tr><td><strong>100</strong></td><td>No Desktop/Laptop</td></tr>
                <tr><td><strong>90</strong></td><td>No ICT equipment</td></tr>
                <tr><td><strong>80</strong></td><td>Only peripherals</td></tr>
                <tr><td><strong>70</strong></td><td>Below-average office coverage</td></tr>
                <tr><td><strong>60</strong></td><td>Computer older than 5 years</td></tr>
                <tr><td><strong>50</strong></td><td>Computer in Poor condition</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('**About**')
        st.markdown("""
        <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:8px; padding:1rem;">
            <p><strong>ICT-AMSOS v2.0</strong></p>
            <p style="font-size:0.85rem; color:#6B7280;">
                AI-Powered ICT Asset Management and Service Optimization System.<br>
                Designed for government agency ICT asset management.
            </p>
        </div>
        """, unsafe_allow_html=True)
