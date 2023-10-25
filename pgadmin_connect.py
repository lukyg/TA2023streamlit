import streamlit as st
import psycopg2

#set koneksi
def init_connection():
    return psycopg2.connect(**st.secrets.connections.postgresql)

conn = init_connection()

cur = conn.cursor()

# Membuat tabel hasil_deteksi
def hasil_deteksi():
    cur.execute("""
        CREATE TABLE IF NOT EXISTS hasil_deteksi(
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL,
            hasil_pembacaan TEXT,
            timestamp TIMESTAMPTZ DEFAULT NOW() -- This adds a timestamp field with the current timestamp
        );
    """)
    conn.commit()