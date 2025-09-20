# Copy dataset
df_clean = df.copy()

# 1. Handle missing data
# Example: drop rows where title or publish_time is missing (if they are critical)
df_clean = df_clean.dropna(subset=['title', 'publish_time'])

# Example: fill missing abstracts with an empty string
df_clean['abstract'] = df_clean['abstract'].fillna('')

# 2. Convert date columns
df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')

# Drop or handle rows where publish_time could not be parsed
df_clean = df_clean.dropna(subset=['publish_time'])

# 3. Extract year for time analysis
df_clean['year'] = df_clean['publish_time'].dt.year

# 4. Create additional useful columns
# e.g., word count of abstract or title
df_clean['title_word_count'] = df_clean['title'].apply(lambda x: len(str(x).split()))
df_clean['abstract_word_count'] = df_clean['abstract'].apply(lambda x: len(str(x).split()))
