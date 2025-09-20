import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    df = pd.read_csv('metadata.csv')
    return df

df = load_data()

st.title("CORD-19 Data Explorer")
st.write("Explore COVID-19 research metadata")

# Sidebar / interactive controls
year_min = int(df['publish_time'].dropna().apply(lambda x: str(x)[:4]).min())
year_max = int(df['publish_time'].dropna().apply(lambda x: str(x)[:4]).max())
selected_years = st.slider("Select publication year range", year_min, year_max, (2020, year_max))

journal_list = df['journal'].dropna().unique().tolist()
selected_journal = st.selectbox("Select a journal (or All)", ["All"] + sorted(journal_list))

# Filter data
df_clean = df.copy()
df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
df_clean = df_clean.dropna(subset=['publish_time'])
df_clean['year'] = df_clean['publish_time'].dt.year

filtered = df_clean[(df_clean['year'] >= selected_years[0]) & (df_clean['year'] <= selected_years[1])]
if selected_journal != "All":
    filtered = filtered[filtered['journal'] == selected_journal]

# Visualizations in app
st.subheader("Publications per Year")
year_counts = filtered['year'].value_counts().sort_index()
fig, ax = plt.subplots()
ax.plot(year_counts.index, year_counts.values, marker='o')
ax.set_xlabel("Year")
ax.set_ylabel("Number of Papers")
st.pyplot(fig)

st.subheader(f"Top Journals in Selected Period")
top_j = filtered['journal'].value_counts().nlargest(5)
fig2, ax2 = plt.subplots()
sns.barplot(x=top_j.values, y=top_j.index, ax=ax2)
ax2.set_xlabel("Number of Papers")
ax2.set_ylabel("Journal")
st.pyplot(fig2)

st.subheader("Sample of Data")
st.dataframe(filtered.head(10))
