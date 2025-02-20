import streamlit as st
import hashlib
import sqlite3
import os

def init_auth():
    # Create users table if it doesn't exist
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users
        (username TEXT PRIMARY KEY,
         password TEXT NOT NULL,
         email TEXT UNIQUE NOT NULL)
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def signup():
    st.subheader("Create New Account")
    new_username = st.text_input("Username*")
    new_password = st.text_input("Password*", type="password")
    new_email = st.text_input("Email*")
    
    if st.button("Sign Up"):
        if not new_username or not new_password or not new_email:
            st.error("Please fill in all required fields")
            return False
        
        hashed_password = hash_password(new_password)
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        try:
            c.execute("INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                    (new_username, hashed_password, new_email))
            conn.commit()
            st.success("Account created successfully! Please log in.")
            return True
        except sqlite3.IntegrityError:
            st.error("Username or email already exists")
            return False
        finally:
            conn.close()
    return False

def login():
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if not username or not password:
            st.error("Please enter both username and password")
            return False
            
        hashed_password = hash_password(password)
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_password))
        result = c.fetchone()
        conn.close()
        
        if result:
            st.success("Logged in successfully!")
            st.session_state['authenticated'] = True
            st.session_state['username'] = username
            return True
        else:
            st.error("Invalid username or password")
            return False
    return False

def is_authenticated():
    return st.session_state.get('authenticated', False)

def logout():
    st.session_state['authenticated'] = False
    st.session_state['username'] = None