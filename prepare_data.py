import pandas as pd
import numpy as np

print("=" * 50)
print("Loading Patrick Zelazko Dataset (2019-2023)...")
print("=" * 50)

# Load the 3M sample from Patrick Zelazko dataset
df = pd.read_csv('flights_sample_3m.csv', low_memory=False)
print(f"✅ Loaded {len(df):,} flights")

# Take a sample for faster processing (500k rows)
df_sample = df.sample(n=500000, random_state=42)
print(f"✅ Sampled {len(df_sample):,} flights for training")

# Rename columns to match our code (Patrick uses different names)
df_sample = df_sample.rename(columns={
    'AIRLINE_CODE': 'AIRLINE',
    'ORIGIN': 'ORIGIN_AIRPORT',
    'DEST': 'DESTINATION_AIRPORT',
    'CRS_DEP_TIME': 'SCHEDULED_DEPARTURE',
    'DEP_DELAY': 'DEPARTURE_DELAY',
    'ARR_DELAY': 'ARRIVAL_DELAY',
    'DELAY_DUE_CARRIER': 'CARRIER_DELAY',
    'DELAY_DUE_WEATHER': 'WEATHER_DELAY',
    'DELAY_DUE_NAS': 'NAS_DELAY',
    'DELAY_DUE_SECURITY': 'SECURITY_DELAY',
    'DELAY_DUE_LATE_AIRCRAFT': 'LATE_AIRCRAFT_DELAY'
})

# Parse date to get year, month, day
df_sample['FL_DATE'] = pd.to_datetime(df_sample['FL_DATE'], format='%Y-%m-%d')
df_sample['YEAR'] = df_sample['FL_DATE'].dt.year
df_sample['MONTH'] = df_sample['FL_DATE'].dt.month
df_sample['DAY'] = df_sample['FL_DATE'].dt.day
df_sample['DAY_OF_WEEK'] = df_sample['FL_DATE'].dt.dayofweek + 1

# Extract hour from scheduled departure (format: 930 = 9:30 AM)
df_sample['DEP_HOUR'] = (df_sample['SCHEDULED_DEPARTURE'] // 100).astype(int)

# Create target: 1 if delayed (≥30 min) or cancelled
df_sample['HIGH_RISK'] = ((df_sample['ARRIVAL_DELAY'] >= 30) | 
                          (df_sample['CANCELLED'] == 1)).astype(int)

# Add season feature
def get_season(month):
    if month in [12, 1, 2]:
        return 0  # Winter
    elif month in [3, 4, 5]:
        return 1  # Spring
    elif month in [6, 7, 8]:
        return 2  # Summer
    else:
        return 3  # Fall

df_sample['SEASON'] = df_sample['MONTH'].apply(get_season)

# Add holiday indicator
def is_holiday(month, day):
    holidays = [(1, 1), (7, 4), (11, 28), (12, 25)]
    for h_month, h_day in holidays:
        if month == h_month and abs(day - h_day) <= 3:
            return 1
    return 0

df_sample['IS_HOLIDAY'] = df_sample.apply(lambda x: is_holiday(x['MONTH'], x['DAY']), axis=1)

# Weekend indicator
df_sample['IS_WEEKEND'] = df_sample['DAY_OF_WEEK'].isin([6, 7]).astype(int)

# Remove rows with missing values
before_drop = len(df_sample)
df_sample = df_sample.dropna(subset=['ARRIVAL_DELAY', 'DEPARTURE_DELAY', 'AIRLINE'])
after_drop = len(df_sample)

print(f"\n📊 Data Processing Complete!")
print(f"   - Rows before: {before_drop:,}")
print(f"   - Rows after: {after_drop:,}")

# Save cleaned data
df_sample.to_csv('flight_data_clean.csv', index=False)
print(f"\n✅ Saved to flight_data_clean.csv")

print(f"\n📈 High risk flights: {df_sample['HIGH_RISK'].sum():,} ({df_sample['HIGH_RISK'].mean()*100:.1f}%)")
print(f"✈️ Airlines: {df_sample['AIRLINE'].nunique()}")
print(f"🏁 Airports: {df_sample['ORIGIN_AIRPORT'].nunique()}")