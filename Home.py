import streamlit as st
import pandas as pd

from pgadmin_connect import conn, cur, hasil_deteksi

hasil_deteksi()

st.title('Plat Nomor Kendaraan OCR')

def get_data_from_db(limit, offset):
    cur.execute(f"SELECT * FROM hasil_deteksi ORDER BY timestamp DESC LIMIT {limit} OFFSET {offset}")
    rows = cur.fetchall()
    return rows

page_number = st.number_input("Halaman", min_value=1, value=1)
rows_per_page = 10
offset = (page_number - 1) * rows_per_page

rows = get_data_from_db(rows_per_page, offset)

st.table(pd.DataFrame(rows, columns=[column.name for column in cur.description]))

cur.execute("SELECT COUNT(*) FROM hasil_deteksi")
total_rows = cur.fetchone()[0]

total_pages = (total_rows + rows_per_page - 1) // rows_per_page

st.write(f"Menampilkan halaman {page_number} dari {total_pages}")
