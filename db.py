# db.py
import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="GaitRecognition",
        user="postgres",
        password="123456"
    )
