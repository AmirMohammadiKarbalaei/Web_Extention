import streamlit as st
import psycopg2
from argon2 import PasswordHasher
import re

# Database credentials
db_credentials = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': '2080869612Amir',
    'host': 'localhost',
    'port': '5432'
}

# Initialize the Argon2id PasswordHasher
ph = PasswordHasher()

# Function to validate email format
def validate_email(email):
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email))

# Function to validate password strength
def validate_password(password):
    return len(password) >= 8

# Hash a password
def hash_password(password):
    return ph.hash(password)

# Check password
def check_password(stored_password, provided_password):
    try:
        ph.verify(stored_password, provided_password)
        return True
    except:
        return False

# Function to handle user sign-up
def signup(email, password):
    if not validate_email(email):
        return False, "Invalid email format."
    elif not validate_password(password):
        return False, "Password should be at least 8 characters long."

    connection = psycopg2.connect(**db_credentials)
    cursor = connection.cursor()
    
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    if cursor.fetchone() is not None:
        cursor.close()
        connection.close()
        return False, "User already exists."

    hashed_password = hash_password(password)
    cursor.execute("INSERT INTO users (email, password) VALUES (%s, %s)", (email, hashed_password))
    connection.commit()
    cursor.close()
    connection.close()
    return True, "User signed up successfully."

# Function to handle user sign-in
def signin(email, password):
    connection = psycopg2.connect(**db_credentials)
    cursor = connection.cursor()
    
    cursor.execute("SELECT password FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    cursor.close()
    connection.close()
    
    if user is None:
        return False, "User does not exist."
    elif not check_password(user[0], password):
        return False, "Incorrect password."
    else:
        return True, "User signed in successfully."


