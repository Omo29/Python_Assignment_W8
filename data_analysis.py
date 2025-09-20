import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# 1. Count papers by publication year
year_counts = df_clean['year'].value_counts().sort_index()
print(year_counts)

# 2. Top journals by paper count
top_journals = df_clean['journal'].value_counts().head(10)
print(top_journals)

# 3. Most frequent words in titles
from collections import Counter
import re

# Preprocess titles
all_titles = df_clean['title'].dropna().astype(str)
words = (all_titles.str.lower()
                     .str.replace(r'[^a-z\s]', ' ', regex=True)
                     .str.split()
                     .explode())
# Remove stop words if you like
stop_words = set(['the','and','of','in','for','on','to','with','a','an','by','is','from'])
words = words[~words.isin(stop_words)]
word_counts = words.value_counts().head(20)
print(word_counts)

# 4. Visualizations

# Publications over time (line chart)
plt.figure(figsize=(10,6))
year_counts.plot(kind='line', marker='o')
plt.title("Number of Publications per Year")
plt.xlabel("Year")
plt.ylabel("Count of Papers")
plt.grid(True)
plt.show()

# Bar chart: top journals
plt.figure(figsize=(10,6))
sns.barplot(y=top_journals.index, x=top_journals.values, palette='viridis')
plt.title("Top 10 Journals Publishing COVID-19 Research")
plt.xlabel("Number of Papers")
plt.ylabel("Journal")
plt.show()

# Histogram: abstract word count
plt.figure(figsize=(10,6))
sns.histplot(df_clean['abstract_word_count'], bins=50, kde=True)
plt.title("Distribution of Abstract Word Count")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()

# Scatter: title word count vs abstract word count
plt.figure(figsize=(8,6))
sns.scatterplot(x='title_word_count', y='abstract_word_count', data=df_clean, alpha=0.6)
plt.title("Title Word Count vs Abstract Word Count")
plt.xlabel("Title Word Count")
plt.ylabel("Abstract Word Count")
plt.show()

# Word cloud of frequent title words
wc = WordCloud(width=800, height=400, background_color='white')
wc.generate(' '.join(all_titles))
plt.figure(figsize=(15,7))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Paper Titles")
plt.show()
