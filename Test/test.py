import streamlit as st
import psycopg2
from argon2 import PasswordHasher
import re
from st_pages import hide_pages,Page

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

    connection = psycopg2.connect(
        dbname=st.secrets["database"]["dbname"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        host=st.secrets["database"]["host"],
        port=st.secrets["database"]["port"]
    )
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
    connection = psycopg2.connect(
        dbname=st.secrets["database"]["dbname"],
        user=st.secrets["database"]["user"],
        password=st.secrets["database"]["password"],
        host=st.secrets["database"]["host"],
        port=st.secrets["database"]["port"]
    )
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
def main():
    st.title("OLALALALA ITS WORKING")
    ## main app loop
    
# Streamlit app
def start():

    hide_pages(
    [
        Page("pages/app.py"),
     ]
    )
    st.title("User Sign In / Sign Up")

    with st.expander("Sign In", expanded=False):
        st.subheader("Sign In to your account")

        # Sign in form
        signin_email = st.text_input("Email", key="signin_email")
        signin_password = st.text_input("Password", type='password', key="signin_password")

        if st.button("Sign In"):
            success, message = signin(signin_email, signin_password)
            if success:
                st.success(message)
                st.switch_page("app.py")
            else:
                st.error(message)

    with st.expander("Sign Up", expanded=False):
        st.subheader("Create a new account")

        # Sign up form
        signup_email = st.text_input("Email", key="signup_email")
        signup_password = st.text_input("Password", type='password', key="signup_password")

        if st.button("Sign Up"):
            if not validate_email(signup_email):
                st.error("Invalid email format.")
            elif not validate_password(signup_password):
                st.error("Password should be at least 8 characters long.")
            else:
                success, message = signup(signup_email, signup_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

if __name__ == '__main__':
    start()
