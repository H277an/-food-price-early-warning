CREATE TABLE IF NOT EXISTS prices (
    date TEXT,
    commodity TEXT,
    index_value REAL
);
CREATE TABLE IF NOT EXISTS weather (
    date TEXT,
    region TEXT,
    precipitation_sum REAL,
    temperature_2m_max REAL
);