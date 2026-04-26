import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import numpy as np


# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title='Supply Chain Analytics Dashboard',
    page_icon='📦',
    layout='wide'
)


# ── Load Data ────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('data/clean_data.csv')


df = load_data()


# ── Load Model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        return pickle.load(f)


model = load_model()


# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title('Navigation')
page = st.sidebar.radio('Go to', ['Overview', 'Analysis', 'Demand Forecaster'])


st.sidebar.markdown('---')
st.sidebar.markdown('**About This Project**')
st.sidebar.markdown(
    'Built by Maximilian Lee \'30\n\n'
    'Tools: Python, pandas, scikit-learn, Streamlit, Plotly'
)


# ══════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == 'Overview':
    st.title('Supply Chain Analytics Dashboard')
    st.markdown('### Key Performance Indicators')


    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Total Revenue', f"${df['revenue_generated'].sum():,.0f}")
    col2.metric('Avg Lead Time', f"{df['lead_times'].mean():.1f} days")
    col3.metric('Total Products Sold', f"{df['number_of_products_sold'].sum():,.0f}")
    col4.metric('Avg Defect Rate', f"{df['defect_rates'].mean():.2%}")


    st.markdown('---')
    st.subheader('Revenue by Product Type')
    rev_by_type = df.groupby('product_type')['revenue_generated'].sum().reset_index()
    fig1 = px.bar(rev_by_type, x='product_type', y='revenue_generated',
                  color='product_type', title='Total Revenue by Product Type',
                  labels={'revenue_generated': 'Revenue ($)', 'product_type': 'Product Type'})
    st.plotly_chart(fig1, use_container_width=True)


    st.subheader('Lead Time vs. Shipping Time Distribution')
    fig2 = px.histogram(df, x='lead_times', nbins=20, color='product_type',
                        title='Distribution of Lead Times',
                        labels={'lead_times': 'Lead Time (days)'})
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2: ANALYSIS
# ══════════════════════════════════════════════════════════════
elif page == 'Analysis':
    st.title('Operations Analysis')


    st.subheader('Shipping Cost vs. Revenue')
    fig3 = px.scatter(df, x='shipping_costs', y='revenue_generated',
                      color='product_type', size='number_of_products_sold',
                      hover_data=['shipping_carriers'],
                      title='Shipping Cost vs Revenue (bubble size = units sold)')
    st.plotly_chart(fig3, use_container_width=True)


    st.subheader('Defect Rate by Supplier')
    defect_by_sup = df.groupby('supplier_name')['defect_rates'].mean().reset_index()
    defect_by_sup = defect_by_sup.sort_values('defect_rates', ascending=False)
    fig4 = px.bar(defect_by_sup, x='supplier_name', y='defect_rates',
                  title='Average Defect Rate by Supplier',
                  labels={'defect_rates': 'Defect Rate', 'supplier_name': 'Supplier'})
    st.plotly_chart(fig4, use_container_width=True)


    st.subheader('Correlation Heatmap')
    num_cols = ['price', 'availability', 'number_of_products_sold',
                'revenue_generated', 'lead_times', 'shipping_times',
                'shipping_costs', 'stock_levels', 'defect_rates']
    corr = df[num_cols].corr()
    fig5 = px.imshow(corr, text_auto=True, aspect='auto',
                     color_continuous_scale='RdBu_r',
                     title='Correlation Matrix of Key Variables')
    st.plotly_chart(fig5, use_container_width=True)


    st.markdown("""
    **How to read the heatmap:** Values close to 1 mean strong positive correlation.
    Values close to -1 mean strong negative correlation. Values near 0 mean no relationship.
    This helps identify which variables most influence demand and revenue.
    """)


# ══════════════════════════════════════════════════════════════
# PAGE 3: DEMAND FORECASTER
# ══════════════════════════════════════════════════════════════
elif page == 'Demand Forecaster':
    st.title('Demand Forecasting Tool')
    st.markdown(
        'Adjust the sliders to simulate different supply chain scenarios.'
        ' The model will predict expected product demand in real time.'
    )


    col1, col2 = st.columns(2)


    with col1:
        price = st.slider('Price ($)', 1.0, 100.0, 25.0)
        availability = st.slider('Availability (%)', 0, 100, 70)
        lead_times = st.slider('Lead Time (days)', 1, 30, 10)


    with col2:
        shipping_times = st.slider('Shipping Time (days)', 1, 10, 3)
        shipping_costs = st.slider('Shipping Cost ($)', 1.0, 50.0, 10.0)
        stock_levels = st.slider('Stock Level (units)', 0, 200, 100)


    input_data = pd.DataFrame({
        'price': [price],
        'availability': [availability],
        'lead_times': [lead_times],
        'shipping_times': [shipping_times],
        'shipping_costs': [shipping_costs],
        'stock_levels': [stock_levels]
    })


    prediction = model.predict(input_data)[0]
    prediction = max(0, prediction)  # Can't be negative


    st.markdown('---')
    st.metric('Predicted Demand (Units)', f'{prediction:.0f} units')


    # Confidence range (approx ± 15%)
    low = prediction * 0.85
    high = prediction * 1.15
    st.info(f'Expected range: {low:.0f} – {high:.0f} units')


    st.markdown('**How to use this tool:**')
    st.markdown(
        '- Try increasing lead time and watch demand drop\n'
        '- Try lowering price and observe the demand increase\n'
        '- This mirrors real IE/ops decisions: how does lead time affect customer demand?'
    )
