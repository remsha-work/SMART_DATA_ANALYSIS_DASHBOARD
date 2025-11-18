import streamlit as st
import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
import tempfile
import io
from pathlib import Path
import hashlib
from typing import Any, Dict, List, Optional, Tuple
import time
import bcrypt
from datetime import datetime
import plotly.graph_objects as go

# Optional libs (used when available)
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import plotly.express as px
except Exception:
    px = None

st.set_page_config(page_title="Smart Data Analysis Dashboard", layout="wide")

# -------------------------
# --- Session state defaults
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "Login"  # start at login
if "files" not in st.session_state:
    st.session_state.files = []  # list of file dicts
if "active_file_idx" not in st.session_state:
    st.session_state.active_file_idx = 0
if "temp_files" not in st.session_state:
    st.session_state.temp_files = []
if "orig_dfs" not in st.session_state:
    st.session_state.orig_dfs = []
if "pages" not in st.session_state:
    st.session_state.pages = [
        "Login",
        "Data Upload",
        "Raw Data", 
        "Data Cleaning",
        "Data Visualization",
        "Report & Export",
        "Hypothesis Summary",
    ]

# Authentication/session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None  # None, "guest", "user"
if "username" not in st.session_state:
    st.session_state.username = None
if "db_conn" not in st.session_state:
    st.session_state.db_conn = None

# Guest limits
if "guest_limits" not in st.session_state:
    st.session_state.guest_limits = {"max_uploads": 3, "max_rows_preview": 500}

# Enhanced CSS injection for better UI
if "css_injected" not in st.session_state:
    st.markdown(
        """
        <style>
        .card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            color: white;
            margin: 20px 0;
        }
        .auth-card {
            background: white;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
            border: 1px solid #e1e5e9;
        }
        .login-title {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
            color: #2c3e50;
            text-align: center;
        }
        .subtitle {
            color: #6c757d;
            font-size: 16px;
            text-align: center;
            margin-bottom: 25px;
        }
        .sidebar-user-info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
        }
        .guest-info {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
        }
        .feature-highlight {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 15px 0;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.css_injected = True
# -------------------------
# --- Helper functions
# -------------------------
def _save_temp_file(uploaded):
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getbuffer())
    tmp.flush()
    tmp.close()
    st.session_state.temp_files.append(tmp.name)
    return tmp.name

def _read_pdf_to_df(path):
    if not pdfplumber:
        raise RuntimeError("pdfplumber not installed.")
    tables = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                tbl = page.extract_table()
                if tbl and len(tbl) > 1:
                    df = pd.DataFrame(tbl[1:], columns=tbl[0])
                    tables.append(df)
            except Exception:
                continue
    if tables:
        return pd.concat(tables, ignore_index=True)
    return None

def _cleanup_temp_files():
    for p in st.session_state.temp_files:
        try:
            os.remove(p)
        except Exception:
            pass
    st.session_state.temp_files = []

def _load_file_to_dict(uploaded):
    name = uploaded.name
    ext = Path(name).suffix.lower()
    file_entry = {
        "name": name,
        "ext": ext,
        "path": None,
        "df": None,
        "raw": uploaded,
        "load_error": None,
    }
    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(uploaded)
                file_entry["df"] = df
            except Exception as e:
                uploaded.seek(0)
                try:
                    df = pd.read_csv(uploaded, engine="python")
                    file_entry["df"] = df
                except Exception as ee:
                    file_entry["load_error"] = f"CSV load error: {ee}"
        elif ext in [".xls", ".xlsx"]:
            try:
                df = pd.read_excel(uploaded)
                file_entry["df"] = df
            except Exception as e:
                file_entry["load_error"] = f"Excel load error: {e}"
        elif ext == ".json":
            try:
                df = pd.read_json(uploaded)
                file_entry["df"] = df
            except Exception as e:
                file_entry["load_error"] = f"JSON load error: {e}"
        elif ext == ".txt":
            try:
                df = pd.read_csv(uploaded, delimiter="\t")
                file_entry["df"] = df
            except Exception:
                try:
                    uploaded.seek(0)
                    df = pd.read_csv(uploaded)
                    file_entry["df"] = df
                except Exception as e:
                    file_entry["load_error"] = f"Text load error: {e}"
        elif ext in [".sqlite", ".db"]:
            tmp_path = _save_temp_file(uploaded)
            file_entry["path"] = tmp_path
            try:
                conn = sqlite3.connect(tmp_path)
                tables = pd.read_sql_query(
                    "SELECT name FROM sqlite_master WHERE type='table';", conn
                )
                file_entry["tables"] = tables["name"].tolist() if not tables.empty else []
                if file_entry.get("tables"):
                    table = file_entry["tables"][0]
                    file_entry["df"] = pd.read_sql_query(f"SELECT * FROM '{table}'", conn)
                    file_entry["loaded_table"] = table
            except Exception as e:
                file_entry["load_error"] = f"SQLite load error: {e}"
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        elif ext == ".pdf":
            tmp_path = _save_temp_file(uploaded)
            file_entry["path"] = tmp_path
            if pdfplumber:
                try:
                    df_pdf = _read_pdf_to_df(tmp_path)
                    if df_pdf is not None:
                        file_entry["df"] = df_pdf
                    else:
                        file_entry["load_error"] = "No tabular data found in PDF."
                except Exception as e:
                    file_entry["load_error"] = f"PDF load error: {e}"
            else:
                file_entry["load_error"] = "pdfplumber not installed; cannot read PDF."
        else:
            file_entry["load_error"] = "Unsupported file type."
    except Exception as e:
        file_entry["load_error"] = str(e)
    return file_entry

def _ensure_orig_dfs_length():
    n_files = len(st.session_state.files)
    while len(st.session_state.orig_dfs) < n_files:
        st.session_state.orig_dfs.append(None)
    if len(st.session_state.orig_dfs) > n_files:
        st.session_state.orig_dfs = st.session_state.orig_dfs[:n_files]

def _generate_hypotheses_for_df(df, name="Active File"):
    if df is None:
        return f"No DataFrame loaded for {name}."
    md = []
    n_rows, n_cols = df.shape
    md.append(f"Dataset: {name} — {n_rows} rows × {n_cols} columns")
    missing = df.isnull().mean().sort_values(ascending=False)
    high_missing = missing[missing > 0.3]
    if not high_missing.empty:
        md.append("Missing data: Some columns have >30% missing values:")
        for col, frac in high_missing.items():
            md.append(f"- {col}: {frac:.0%} missing")
    else:
        md.append("- Missingness looks low.")
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] >= 2:
        corr = num.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
        top_corr = [(a, b, v) for (a, b), v in corr.items() if a != b][:5]
        if top_corr:
            md.append("Potential relationships (top correlations):")
            for a, b, v in top_corr:
                md.append(f"- {a} vs {b} — correlation {v:.2f}")
    else:
        md.append("- Not enough numeric columns to compute correlations.")
    cat = df.select_dtypes(include=["object", "category"])
    if not cat.empty:
        md.append("Categorical columns (sample):")
        for c in cat.columns[:5]:
            md.append(f"- {c}: {df[c].nunique(dropna=True)} unique values")
    return "\n".join(md)


client = MongoClient("mongodb://localhost:27017/")
db = client["smart_data_analysis"]
users_collection = db["user_info"]


# Insert a test user
# users_collection.insert_one({
#     "username": "admin",
#     "password_hash": "12345",
#     "full_name": "Admin User",
#     "email": "admin@example.com",
#     "created_at": pd.Timestamp.now(),
#     "last_login": None
# })

def create_user_mongo(username, password, full_name="", email=""):
    if not username or not password:
        return False, "Username and password are required."

    # check exists first (optional because index + DuplicateKeyError will handle duplicates)
    if users_collection.find_one({"username": username}):
        return False, "Username already exists."

    # hash password
    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    hashed_pw_str = hashed_pw.decode("utf-8")  # convert bytes to string


    user_doc = {
        "username": username,
        "password": hashed_pw.decode("utf-8"),  # store as string
        "full_name": full_name,
        "email": email,
    }

    try:
        users_collection.insert_one(user_doc)
    except DuplicateKeyError:
        return False, "Username already exists (DB constraint)."

    return True, "User created successfully."


def verify_user_mongo(username, password):
    user = users_collection.find_one({"username": username})
    if not user:
        return False, "No such user."

    stored_hash = user.get("password", "").encode("utf-8")
    if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
        return True, user
    return False, "Invalid password."



def get_user_info(username):
    try:
        user = users_collection.find_one({"username": username}, {"_id": 0})
        return user
    except Exception as e:
        st.error(f"Error fetching user info: {e}")
        return None

def update_user_profile(username, full_name=None, email=None):
    """Update user profile information in MongoDB"""
    try:
        result = users_collection.update_one(
            {"username": username},
            {"$set": {"full_name": full_name, "email": email}}
        )
        if result.matched_count > 0:
            return True, "Profile updated successfully"
        else:
            return False, "User not found"
    except Exception as e:
        return False, f"Error updating profile: {e}"

def list_all_users():
    """Admin function to list all users from MongoDB"""
    try:
        users = list(users_collection.find({}, {"_id": 0}))
        return users
    except Exception as e:
        st.error(f"Error fetching users: {e}")
        return []


def change_password(username, old_password, new_password):
    """Allow user to change their password in MongoDB"""
    try:
        user = users_collection.find_one({"username": username})
        if not user:
            return False, "User not found"

        # verify old password (assuming stored as hash string)
        if not _verify_password(old_password, user["password_hash"]):
            return False, "Current password is incorrect"

        new_hash = _hash_password(new_password)
        users_collection.update_one(
            {"username": username},
            {"$set": {"password_hash": new_hash}}
        )
        return True, "Password changed successfully"
    except Exception as e:
        return False, f"Error changing password: {e}"

# --- Local authentication (in-memory for development) ---
def _hash_password(password: str, salt: str = "change_this_salt") -> str:
    h = hashlib.sha256()
    h.update((salt + password).encode("utf-8"))
    return h.hexdigest()

if "local_users" not in st.session_state:
    st.session_state.local_users = {}


def create_user_mongo(username, password, full_name=None, email=None):
    """Create user in MongoDB"""
    if users_collection.find_one({"username": username}):
        return False, "User already exists"
    
    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    users_collection.insert_one({
        "username": username,
        "password_hash": pw_hash,
        "full_name": full_name or "",
        "email": email or "",
        "created_at": datetime.utcnow(),
        "last_login": None
    })
    return True, "Account created successfully"


def verify_user_mongo(username, password):
    """Verify user credentials from MongoDB"""
    user = users_collection.find_one({"username": username})
    if not user:
        return False, "User not found"
    
    stored_hash = user["password_hash"].encode("utf-8")
    if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
        # Update last login
        users_collection.update_one(
            {"username": username},
            {"$set": {"last_login": datetime.utcnow()}}
        )
        return True, {
            "username": user["username"],
            "full_name": user.get("full_name"),
            "email": user.get("email")
        }
    else:
        return False, "Invalid password"


# Keep original functions as fallback
def create_user_local(username, password, full_name=None, email=None):
    if username in st.session_state.local_users:
        return False, "User already exists"
    st.session_state.local_users[username] = {
        "password_hash": _hash_password(password),
        "full_name": full_name,
        "email": email,
        "created_at": time.time(),
    }
    return True, "Account created successfully"

def verify_user_local(username, password):
    row = st.session_state.local_users.get(username)
    if not row:
        return False, "User not found"
    pw_hash = _hash_password(password)
    if pw_hash == row["password_hash"]:
        return True, {"username": username, "full_name": row.get("full_name"), "email": row.get("email")}
    return False, "Invalid password"

# Role-aware gate
def require_login_ui(require_login: bool = False) -> bool:
    if st.session_state.get("logged_in") and st.session_state.get("user_role") == "user":
        return True
    if st.session_state.get("user_role") == "guest" and not require_login:
        return True
    if st.session_state.get("user_role") == "guest" and require_login:
        st.warning("This feature requires an account. Please register or login to continue.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Login / Register"):
                st.session_state.page = "Login"
                st.rerun()
        with c2:
            if st.button("Create account"):
                st.session_state.page = "Login"
                st.rerun()
        return False
    st.warning("You are not logged in. Continue as guest or sign up for full features.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Continue as guest"):
            st.session_state.user_role = "guest"
            st.session_state.username = "Guest"
            st.session_state.logged_in = False
            st.session_state.page = "Data Upload"
            st.success("Continuing as Guest — redirected to Data Upload.")
            st.rerun()
    with c2:
        if st.button("Go to Login / Register"):
            st.session_state.page = "Login"
            st.rerun()
    return False

def logout_user():
    """Handle user logout"""
    st.session_state.logged_in = False
    st.session_state.user_role = None
    st.session_state.username = None
    conn = st.session_state.get("db_conn")
    try:
        if conn:
            conn.close()
    except Exception:
        pass
    st.session_state.db_conn = None
    st.session_state.page = "Login"

# -------------------------
# --- IMPROVED SIDEBAR with Login/Logout fix
# -------------------------
with st.sidebar:
    st.title("📊 Dashboard Panel")

    # User info section
    if st.session_state.user_role == "user" and st.session_state.username:
        st.markdown(
            f"""
            <div class="sidebar-user-info">
                <div style="font-size: 18px; font-weight: 600;">👋 Welcome!</div>
                <div style="font-size: 16px; margin: 5px 0;">{st.session_state.username}</div>
                <div style="font-size: 12px; opacity: 0.8;">Logged in as User</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif st.session_state.user_role == "guest":
        st.markdown(
            """
            <div class="guest-info">
                <div style="font-size: 18px; font-weight: 600;">🎭 Guest Mode</div>
                <div style="font-size: 14px; opacity: 0.8;">Limited features available</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Navigation")

    # Hide "Login" from navigation when authenticated
    pages_to_show = list(st.session_state.pages)
    if st.session_state.user_role == "user":
        pages_to_show = [p for p in pages_to_show if p != "Login"]

    for p in pages_to_show:
        if st.button(p, use_container_width=True, key=f"nav_{p}"):
            st.session_state.page = p
            st.rerun()

    st.markdown("---")

    # Files section
    st.write("Files uploaded")
    if st.session_state.files:
        names = [f["name"] for f in st.session_state.files]
        idx = st.selectbox(
            "Active file",
            options=list(range(len(names))),
            format_func=lambda i: f"{i+1}. {names[i]}",
            index=st.session_state.active_file_idx,
        )
        st.session_state.active_file_idx = idx
    else:
        st.info("No files uploaded yet. Go to Data Upload to add files.")

    st.markdown("---")

    # Contextual auth action
    if st.session_state.user_role == "guest":
        st.info("You are using the app as a Guest. Some features are disabled.")
        if st.button("Create account / Login", use_container_width=True, key="sidebar_go_login"):
            st.session_state.page = "Login"
            st.rerun()
    elif st.session_state.user_role == "user":
        if st.button("🚪 Logout", use_container_width=True, type="secondary", key="sidebar_logout"):
            logout_user()
            st.rerun()
    else:
        # not logged in (no role set)
        if st.button("Continue as guest (quick)", use_container_width=True, key="sidebar_guest"):
            st.session_state.user_role = "guest"
            st.session_state.username = "Guest"
            st.session_state.logged_in = False
            st.session_state.page = "Data Upload"
            st.rerun()

# -------------------------
# --- Page: SIMPLIFIED Login
# -------------------------
if st.session_state.page == "Login":
    st.title("🔐 Welcome to Smart Data Analysis Dashboard")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # --- Card Layout ---
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">Get Started</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="subtitle">Create an account or login to access all features</div>',
            unsafe_allow_html=True,
        )

        # --- Toggle between Login and Register ---
        auth_mode = st.radio(
            "Authentication Mode",
            ["Login", "Register"],
            horizontal=True,
            label_visibility="collapsed",
        )

        # -----------------------
        # --- LOGIN MODE ---
        # -----------------------
        if auth_mode == "Login":
            with st.form("login_form"):
                st.subheader("🔑 Login to Your Account")
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("Login", use_container_width=True, type="primary")

                if submit:
                    if not username or not password:
                        st.error("Please enter both username and password")
                    else:
                        ok, info = verify_user_mongo(username, password)
                        
                        if ok:
                            st.session_state.logged_in = True
                            st.session_state.user_role = "user"
                            st.session_state.username = info.get("username") if isinstance(info, dict) else username
                            st.success(f"Welcome back, {st.session_state.username}!")
                            time.sleep(0.5)
                            st.session_state.page = "Data Upload"
                            st.rerun()
                        else:
                            st.error(f"Login failed: {info}")

        # -----------------------
        # --- REGISTER MODE ---
        # -----------------------
        else:
            with st.form("register_form"):
                st.subheader("📝 Create New Account")
                username = st.text_input("Choose Username", placeholder="Enter a unique username")
                password = st.text_input("Choose Password", type="password", placeholder="Create a strong password")
                full_name = st.text_input("Full Name (Optional)", placeholder="Your full name")
                email = st.text_input("Email (Optional)", placeholder="your.email@example.com")
                submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")

                if submit:
                    if not username or not password:
                        st.error("Please provide username and password")
                    elif len(username) < 3:
                        st.error("Username must be at least 3 characters long")
                    elif len(password) < 4:
                        st.error("Password must be at least 4 characters long")
                    else:
                        ok, msg = create_user_mongo(username, password, full_name, email)
                        if ok:
                            st.session_state.logged_in = True
                            st.session_state.user_role = "user"
                            st.session_state.username = username
                            st.success(f"Account created! Welcome, {username}!")
                            time.sleep(0.5)
                            st.session_state.page = "Data Upload"
                            st.rerun()
                        else:
                            st.error(f"Registration failed: {msg}")

        # -----------------------
        # --- Feature Highlights ---
        # -----------------------
        st.markdown("---")
        st.markdown(
            """
            <div class="feature-highlight">
                <h4 style="margin: 0 0 10px 0;">✨ What you get with an account:</h4>
                <div style="font-size: 14px;">
                    • Upload unlimited files<br>
                    • Connect to SQL databases<br>
                    • Advanced data cleaning tools<br>
                    • Export processed data<br>
                    • Full data visualization suite
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -----------------------
        # --- Guest Access Option ---
        # -----------------------
        st.markdown("🎭 Or try as *Guest*")
        st.markdown("Quick access with limited features (up to 3 file uploads)")

        if st.button("Continue as Guest", use_container_width=True, type="secondary"):
            st.session_state.user_role = "guest"
            st.session_state.username = "Guest"
            st.session_state.logged_in = False
            st.session_state.page = "Data Upload"
            st.success("Continuing as Guest...")
            time.sleep(0.3)
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

# # -------------------------
# # --- Page: Data Upload with SQL Connection
# # -------------------------
# elif st.session_state.page == "Data Upload":
#     if not require_login_ui(require_login=False):
#         st.stop()

#     st.title("📂 Data Upload & Connection Hub")

#     # Toggle between File Upload and SQL Connection
#     upload_mode = st.radio("Choose data source:", ["📁 Upload Files", "🗄 Connect to Database"], horizontal=True)

#     # -------------------------
#     # --- File Upload Section
#     # -------------------------
#     if upload_mode == "📁 Upload Files":
#         st.subheader("Upload Your Data File(s)")

#         # Guest upload limit
#         if st.session_state.user_role == "guest":
#             max_u = st.session_state.guest_limits["max_uploads"]
#             if len(st.session_state.files) >= max_u:
#                 st.warning(f"As a guest you can upload up to {max_u} files. Create an account to upload more.")
#                 st.info("Use the sidebar to go to Login to register.")
#                 uploaded_files = None
#             else:
#                 uploaded_files = st.file_uploader(
#                     "Choose file(s)",
#                     type=["csv", "xls", "xlsx", "json", "txt", "sqlite", "db", "pdf"],
#                     accept_multiple_files=True,
#                 )
#         else:
#             uploaded_files = st.file_uploader(
#                 "Choose file(s)",
#                 type=["csv", "xls", "xlsx", "json", "txt", "sqlite", "db", "pdf"],
#                 accept_multiple_files=True,
#             )

#         if uploaded_files:
#             added = 0
#             for uploaded in uploaded_files:
#                 already = any(
#                     (f["name"] == uploaded.name and getattr(f.get("raw"), "size", None) == uploaded.size)
#                     for f in st.session_state.files
#                 )
#                 if already:
#                     continue
#                 entry = _load_file_to_dict(uploaded)
#                 st.session_state.files.append(entry)
#                 added += 1
#             _ensure_orig_dfs_length()
#             if added:
#                 st.success(f"Added {added} file(s). Use the sidebar to select active file.")
#             else:
#                 st.info("No new files added (duplicates were skipped).")

#     # -------------------------
#     # --- MySQL Connection Section
#     # -------------------------
#     else:  # Database Connection (MySQL only)
#         st.subheader("🗄 Connect to MySQL Database")

#         host = st.text_input("Host", placeholder="eg., sql12.freesqldatabase.com")
#         port = st.number_input("Port", value=3306)
#         database = st.text_input("Database Name")
#         username = st.text_input("Username", key="db_user")
#         password = st.text_input("Password", type="password", key="db_pass")

#         import mysql.connector
#         import pandas as pd

#         def get_mysql_connection(host, port, user, password, database):
#             return mysql.connector.connect(
#                 host=host,
#                 port=int(port),
#                 user=user,
#                 password=password,
#                 database=database
#             )

#         if st.button("🔌 Connect to Database"):
#             try:
#                 conn = get_mysql_connection(host, port, username, password, database)
#                 cursor = conn.cursor()
#                 cursor.execute("SHOW TABLES;")
#                 tables = [table[0] for table in cursor.fetchall()]
#                 cursor.close()

#                 if tables:
#                     st.success(f"Connected successfully! Found {len(tables)} tables.")
#                     selected_table = st.selectbox("Select Table to Load", tables)

#                     if st.button("Load Table into Dashboard"):
#                         df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)

#                         # Add to session_state exactly like other uploaded files
#                         if "files" not in st.session_state:
#                             st.session_state.files = []

#                         st.session_state.files.append({
#                             "name": f"{database}.{selected_table}",
#                             "ext": ".sql",
#                             "path": None,
#                             "df": df,
#                             "raw": None,
#                             "load_error": None,
#                             "source": "mysql"
#                         })

#                         st.success(f"Table '{selected_table}' loaded! You can now access it in Raw Data and other pages.")
#                         st.rerun()  # rerun to refresh Raw Data page
#                 else:
#                     st.info("No tables found in the database.")

#                 conn.close()
#             except Exception as e:
#                 st.error(f"Database connection failed: {e}")

#     # -------------------------
#     # --- Remove Files Section
#     # -------------------------
#     if "files" in st.session_state and st.session_state.files:
#         st.subheader("🗑 Remove Uploaded / Loaded Files")

#         file_names = [f["name"] for f in st.session_state.files]
#         files_to_remove = st.multiselect("Select file(s) to remove:", file_names)

#         if st.button("Remove Selected File(s)"):
#             if files_to_remove:
#                 st.session_state.files = [f for f in st.session_state.files if f["name"] not in files_to_remove]
#                 st.success(f"Removed {len(files_to_remove)} file(s).")
#                 st.rerun()
#             else:
#                 st.warning("Please select at least one file to remove.")

#     # -------------------------
#     # --- Next Button
#     # -------------------------
#     if st.button("Next: Explore Data →", type="primary"):
#         if st.session_state.files:
#             st.session_state.page = "Raw Data"
#             st.rerun()
#         else:
#             st.warning("Please upload files or connect to a database first!")

# -------------------------
# --- Page: Data Upload with SQL Connection
# -------------------------
elif st.session_state.page == "Data Upload":
    if not require_login_ui(require_login=False):
        st.stop()

    st.title("📂 Data Upload & Connection Hub")

    # Toggle between File Upload and SQL Connection
    upload_mode = st.radio("Choose data source:", ["📁 Upload Files", "🗄 Connect to Database"], horizontal=True)

    # -------------------------
    # --- FILE UPLOAD MODE
    # -------------------------
    if upload_mode == "📁 Upload Files":
        st.subheader("Upload Your Data File(s)")

        # Handle guest upload limit
        if st.session_state.user_role == "guest":
            max_u = st.session_state.guest_limits["max_uploads"]
            if len(st.session_state.files) >= max_u:
                st.warning(f"As a guest you can upload up to {max_u} files. Create an account to upload more.")
                st.info("Use the sidebar to go to Login to register.")
                uploaded_files = None
            else:
                uploaded_files = st.file_uploader(
                    "Choose file(s)",
                    type=["csv", "xls", "xlsx", "json", "txt", "sqlite", "db", "pdf"],
                    accept_multiple_files=True,
                )
        else:
            uploaded_files = st.file_uploader(
                "Choose file(s)",
                type=["csv", "xls", "xlsx", "json", "txt", "sqlite", "db", "pdf"],
                accept_multiple_files=True,
            )

        # Add uploaded files to session_state
        if uploaded_files:
            added = 0
            for uploaded in uploaded_files:
                already = any(
                    (f["name"] == uploaded.name and getattr(f.get("raw"), "size", None) == uploaded.size)
                    for f in st.session_state.files
                )
                if already:
                    continue
                entry = _load_file_to_dict(uploaded)
                st.session_state.files.append(entry)
                added += 1
            _ensure_orig_dfs_length()
            if added:
                st.success(f"✅ Added {added} file(s). Use the sidebar to select active file.")
            else:
                st.info("No new files added (duplicates were skipped).")

        # Option to remove uploaded files
        if st.session_state.files:
            st.subheader("Manage Uploaded Files")
            file_names = [f["name"] for f in st.session_state.files]
            file_to_remove = st.selectbox("Select file to remove", ["-- Select --"] + file_names)
            if file_to_remove != "-- Select --":
                if st.button("🗑 Remove Selected File"):
                    st.session_state.files = [f for f in st.session_state.files if f["name"] != file_to_remove]
                    st.success(f"Removed file: {file_to_remove}")
                    st.rerun()

    # -------------------------
    # --- SQL DASHBOARD LAUNCH ONLY
    # -------------------------
    else:
        st.subheader("🗄 Connect to MySQL Database")
        st.info("Click below to launch the dedicated SQL Data Dashboard.")

        if st.button("🚀 Launch SQL Data Dashboard"):
            import subprocess
            try:
                subprocess.Popen(["streamlit", "run", "sql.py", "--server.port=8502"], shell=True)
                st.success("✅ SQL Data Dashboard launched in a new browser tab (port 8502).")
                st.stop()
            except Exception as e:
                st.error(f"❌ Failed to launch SQL Dashboard: {e}")

    # -------------------------
    # --- NEXT PAGE BUTTON
    # -------------------------
    if st.button("Next: Explore Data →", type="primary"):
        if st.session_state.files:
            st.session_state.page = "Raw Data"
            st.rerun()
        else:
            st.warning("Please upload files or connect to a database first!")



# -------------------------
# --- Page: Raw Data Explorer
# -------------------------
elif st.session_state.page == "Raw Data":
    if not require_login_ui(require_login=False):
        st.stop()

    st.title("📜 Raw Data Explorer")
    if not st.session_state.files:
        st.warning("Upload a file or import a SQL table first")
    else:
        compare = st.checkbox("Compare first two files side-by-side", value=False)
        if compare and len(st.session_state.files) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                st.header(f"File 1: {st.session_state.files[0]['name']}")
                df1 = st.session_state.files[0].get("df")
                if df1 is not None:
                    display_df = (
                        df1.head(st.session_state.guest_limits["max_rows_preview"])
                        if st.session_state.user_role == "guest"
                        else df1.head()
                    )
                    st.write(display_df)
                    st.dataframe(display_df)
                else:
                    st.info("No dataframe loaded.")
            with col2:
                st.header(f"File 2: {st.session_state.files[1]['name']}")
                df2 = st.session_state.files[1].get("df")
                if df2 is not None:
                    display_df = (
                        df2.head(st.session_state.guest_limits["max_rows_preview"])
                        if st.session_state.user_role == "guest"
                        else df2.head()
                    )
                    st.write(display_df)
                    st.dataframe(display_df)
                else:
                    st.info("No dataframe loaded.")
        else:
            active = st.session_state.files[st.session_state.active_file_idx]
            st.subheader(f"Active file: {active['name']}")
            df = active.get("df")
            if df is not None:
                max_rows = (
                    st.session_state.guest_limits["max_rows_preview"]
                    if st.session_state.user_role == "guest"
                    else None
                )
                if max_rows:
                    st.info(f"Guest preview limited to {max_rows} rows.")
                    st.write(df.head(max_rows))
                    st.dataframe(df.head(max_rows))
                else:
                    st.write(df.head())
                    st.dataframe(df)
                
                cols = df.columns.tolist()
                with st.expander("Preview & Controls", expanded=True):
                    view_cols = st.multiselect("Columns to display", cols, default=cols)
                    if view_cols:
                        st.dataframe(df[view_cols].head(max_rows) if max_rows else df[view_cols])
                
                if st.checkbox("Enable quick filter"):
                    filter_col = st.selectbox("Filter column", cols)
                    op = st.selectbox("Operator", ["==", "!=", "contains", ">", "<", ">=", "<="])
                    val = st.text_input("Value to compare")
                    if st.button("Apply filter"):
                        try:
                            df2 = df.copy()
                            if op == "contains":
                                df2 = df2[df2[filter_col].astype(str).str.contains(val, na=False, case=False)]
                            else:
                                try:
                                    num = float(val)
                                    if op == "==":
                                        df2 = df2[df2[filter_col] == num]
                                    elif op == "!=":
                                        df2 = df2[df2[filter_col] != num]
                                    elif op == ">":
                                        df2 = df2[df2[filter_col] > num]
                                    elif op == "<":
                                        df2 = df2[df2[filter_col] < num]
                                    elif op == ">=":
                                        df2 = df2[df2[filter_col] >= num]
                                    elif op == "<=":
                                        df2 = df2[df2[filter_col] <= num]
                                except ValueError:
                                    if op == "==":
                                        df2 = df2[df2[filter_col].astype(str) == val]
                                    elif op == "!=":
                                        df2 = df2[df2[filter_col].astype(str) != val]
                            st.write(f"Filtered rows: {len(df2)}")
                            st.dataframe(df2.head(max_rows) if max_rows else df2)
                            st.session_state.files[st.session_state.active_file_idx]["df"] = df2
                        except Exception as e:
                            st.error(f"Could not apply filter: {e}")
                
                if st.checkbox("Enable sorting"):
                    sort_cols = st.multiselect("Columns to sort by", cols)
                    asc = st.radio("Order", ("Ascending", "Descending"))
                    if st.button("Apply sort") and sort_cols:
                        df_sorted = df.sort_values(by=sort_cols, ascending=(asc == "Ascending"))
                        st.session_state.files[st.session_state.active_file_idx]["df"] = df_sorted
                        st.success("Sorted")
                        st.rerun()
            else:
                st.info("No DataFrame available for this file.")

    if st.button("Next: Data Cleaning →", type="primary"):
        st.session_state.page = "Data Cleaning"
        st.rerun()

# -------------------------
# --- Page: Data Cleaning 
# -------------------------
elif st.session_state.page == "Data Cleaning":
    if not require_login_ui(require_login=False):
        st.stop()

    st.title("🧹 Data Cleaning")
    if not st.session_state.files:
        st.warning("Upload files first in Data Upload.")
    else:
        active = st.session_state.files[st.session_state.active_file_idx]
        df = active.get("df")
        if df is not None:
            st.subheader(f"Cleaning: {active['name']}")
            
            # Basic data info
            st.write("Dataset Info:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # --------------------------------
            # Missing data handling
            # --------------------------------
            st.subheader("Missing Data Handling")
            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                st.write(f"Columns with missing data: {missing_cols}")
                for col in missing_cols:
                    missing_pct = (df[col].isnull().sum() / len(df)) * 100
                    st.write(f"- {col}: {missing_pct:.1f}% missing")
                
                action = st.selectbox("Choose action for missing data:", 
                                    ["Keep as is", 
                                     "Drop rows with any missing", 
                                     "Drop columns with >50% missing", 
                                     "Fill with mean/mode"])
                
                if st.button("Apply Missing Data Action"):
                    if action == "Drop rows with any missing":
                        df_cleaned = df.dropna()
                        st.session_state.files[st.session_state.active_file_idx]["df"] = df_cleaned
                        st.success(f"Dropped rows. New shape: {df_cleaned.shape}")
                    elif action == "Drop columns with >50% missing":
                        thresh = len(df) * 0.5
                        df_cleaned = df.dropna(axis=1, thresh=thresh)
                        st.session_state.files[st.session_state.active_file_idx]["df"] = df_cleaned
                        st.success(f"Dropped columns. New shape: {df_cleaned.shape}")
                    elif action == "Fill with mean/mode":
                        df_filled = df.copy()
                        for col in missing_cols:
                            if df[col].dtype in ['int64', 'float64']:
                                df_filled[col].fillna(df[col].mean(), inplace=True)
                            else:
                                df_filled[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
                        st.session_state.files[st.session_state.active_file_idx]["df"] = df_filled
                        st.success("Filled missing values")
                    st.rerun()
            else:
                st.info("No missing data found!")
            
            # --------------------------------
            # Duplicate handling
            # --------------------------------
            st.subheader("Duplicate Handling")
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                st.write(f"Found {duplicates} duplicate rows")
                if st.button("Remove Duplicates"):
                    df_dedup = df.drop_duplicates()
                    st.session_state.files[st.session_state.active_file_idx]["df"] = df_dedup
                    st.success(f"Removed {duplicates} duplicates")
                    st.rerun()
            else:
                st.info("No duplicates found!")

            # --------------------------------
            # NEW FEATURE: Drop Rows/Columns
            # --------------------------------
            st.subheader("Drop Rows/Columns")
            drop_option = st.radio("Choose what to drop:", ["Row(s)", "Column(s)"])
            if drop_option == "Row(s)":
                rows_to_drop = st.multiselect("Select rows to drop (by index):", df.index.tolist())
                if st.button("Drop Selected Rows"):
                    df_dropped = df.drop(rows_to_drop, axis=0)
                    st.session_state.files[st.session_state.active_file_idx]["df"] = df_dropped
                    st.success(f"Dropped {len(rows_to_drop)} row(s)")
                    st.rerun()
            else:
                cols_to_drop = st.multiselect("Select columns to drop:", df.columns.tolist())
                if st.button("Drop Selected Columns"):
                    df_dropped = df.drop(cols_to_drop, axis=1)
                    st.session_state.files[st.session_state.active_file_idx]["df"] = df_dropped
                    st.success(f"Dropped {len(cols_to_drop)} column(s)")
                    st.rerun()

            # --------------------------------
            # NEW FEATURE: Direct Data Editing
            # --------------------------------
            st.subheader("Edit Data")
            st.info("You can edit values directly in the table below:")
            edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)
            if st.button("Save Edited Data"):
                st.session_state.files[st.session_state.active_file_idx]["df"] = edited_df
                st.success("Edits saved successfully!")
                st.rerun()

            # --------------------------------
            # EXTRA CLEANING FEATURES
            # --------------------------------
            st.subheader("Additional Cleaning Options")
            with st.expander("Standardize Column Names"):
                if st.button("Make Columns Lowercase & Replace Spaces with _"):
                    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
                    st.session_state.files[st.session_state.active_file_idx]["df"] = df
                    st.success("Standardized column names")
                    st.rerun()

            with st.expander("Convert Data Types"):
                col_to_convert = st.selectbox("Select column to convert:", df.columns)
                dtype = st.selectbox("Convert to:", ["int", "float", "string", "datetime"])
                if st.button("Apply Conversion"):
                    df_converted = df.copy()
                    try:
                        if dtype == "int":
                            df_converted[col_to_convert] = df_converted[col_to_convert].astype(int)
                        elif dtype == "float":
                            df_converted[col_to_convert] = df_converted[col_to_convert].astype(float)
                        elif dtype == "string":
                            df_converted[col_to_convert] = df_converted[col_to_convert].astype(str)
                        elif dtype == "datetime":
                            df_converted[col_to_convert] = pd.to_datetime(df_converted[col_to_convert], errors="coerce")
                        st.session_state.files[st.session_state.active_file_idx]["df"] = df_converted
                        st.success(f"Converted {col_to_convert} to {dtype}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Conversion failed: {e}")

        else:
            st.info("No DataFrame loaded for active file.")

    if st.button("Next: Visualization →", type="primary"):
        st.session_state.page = "Data Visualization" 
        st.rerun()


# -------------------------
# --- Page: Data Visualization
# -------------------------
elif st.session_state.page == "Data Visualization":
    st.title("📈 Data Visualization")

    if not st.session_state.files:
        st.warning("⚠ No files uploaded yet. Please upload at least one file.")
    else:
        files = st.session_state.files
        file_names = [f["name"] for f in files]

        # --- Visualization Mode Selection ---
        mode = st.radio(
            "Select Visualization Mode",
            ["📊 View Single File", "🧩 Compare Side-by-Side", "🔄 Cross Compare (X from one, Y from another)"]
        )

        # -------------------------
        # --- MODE 1: Single File
        # -------------------------
        if mode == "📊 View Single File":
            file_choice = st.selectbox("Select file", file_names, index=st.session_state.active_file_idx)
            active = next(f for f in files if f["name"] == file_choice)
            df = active.get("df")

            if df is not None and not df.empty:
                st.subheader(f"📂 Active file: {file_choice}")
                st.write("Preview:", df.head())

                chart_type = st.selectbox(
                    "Chart type", ["None", "Line", "Bar", "Histogram", "Boxplot", "Scatter", "Pie"]
                )

                if chart_type != "None":
                    cols = df.columns.tolist()

                    if chart_type in ["Line", "Bar", "Scatter"]:
                        x = st.selectbox("X axis", cols)
                        y = st.selectbox("Y axis", cols)
                        if chart_type == "Line":
                            fig = px.line(df, x=x, y=y)
                        elif chart_type == "Bar":
                            fig = px.bar(df, x=x, y=y)
                        else:
                            fig = px.scatter(df, x=x, y=y)
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "Histogram":
                        col = st.selectbox("Column", cols)
                        fig = px.histogram(df, x=col)
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "Boxplot":
                        col = st.selectbox("Column", cols)
                        fig = px.box(df, y=col)
                        st.plotly_chart(fig, use_container_width=True)

                    elif chart_type == "Pie":
                        labels = st.selectbox("Labels column", cols)
                        values = st.selectbox("Values column", cols)
                        fig = px.pie(df, names=labels, values=values)
                        st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("No valid DataFrame found for this file.")

        # -------------------------
        # --- MODE 2: Side-by-Side Comparison
        # -------------------------
        elif mode == "🧩 Compare Side-by-Side":
            if len(files) < 2:
                st.warning("⚠ You need at least two uploaded files to compare.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    file1_name = st.selectbox("Select File 1", file_names, key="compare_file1")
                with col2:
                    file2_name = st.selectbox("Select File 2", [f for f in file_names if f != file1_name], key="compare_file2")

                df1 = next(f for f in files if f["name"] == file1_name).get("df")
                df2 = next(f for f in files if f["name"] == file2_name).get("df")

                chart_type = st.selectbox("Chart type", ["Line", "Bar", "Histogram", "Boxplot", "Scatter", "Pie"])

                st.markdown("### 📈 Side-by-Side Comparison")
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown(f"#### {file1_name}")
                    cols1 = df1.columns.tolist()
                    x1 = st.selectbox("X axis", cols1, key="x1_side")
                    y1 = st.selectbox("Y axis", cols1, key="y1_side")
                    fig1 = px.line(df1, x=x1, y=y1) if chart_type == "Line" else (
                           px.bar(df1, x=x1, y=y1) if chart_type == "Bar" else (
                           px.scatter(df1, x=x1, y=y1) if chart_type == "Scatter" else None))
                    if fig1:
                        st.plotly_chart(fig1, use_container_width=True)

                with c2:
                    st.markdown(f"#### {file2_name}")
                    cols2 = df2.columns.tolist()
                    x2 = st.selectbox("X axis", cols2, key="x2_side")
                    y2 = st.selectbox("Y axis", cols2, key="y2_side")
                    fig2 = px.line(df2, x=x2, y=y2) if chart_type == "Line" else (
                           px.bar(df2, x=x2, y=y2) if chart_type == "Bar" else (
                           px.scatter(df2, x=x2, y=y2) if chart_type == "Scatter" else None))
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True)

        # -------------------------
        # --- MODE 3: Cross Comparison (X from one, Y from another)
        # -------------------------
        elif mode == "🔄 Cross Compare (X from one, Y from another)":
            if len(files) < 2:
                st.warning("⚠ You need at least two uploaded files to cross-compare.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    x_file = st.selectbox("Select file for X-axis", file_names, key="x_file_cross")
                with col2:
                    y_file = st.selectbox("Select file for Y-axis", [f for f in file_names if f != x_file], key="y_file_cross")

                df_x = next(f for f in files if f["name"] == x_file).get("df")
                df_y = next(f for f in files if f["name"] == y_file).get("df")

                x_col = st.selectbox(f"Select X-axis column from {x_file}", df_x.columns.tolist(), key="x_col_cross")
                y_col = st.selectbox(f"Select Y-axis column from {y_file}", df_y.columns.tolist(), key="y_col_cross")

                chart_type = st.selectbox("Chart type", ["Line", "Bar", "Scatter"])

                st.markdown(f"### 🔄 Cross Comparison: {x_file} (X) vs {y_file} (Y)")

                fig = go.Figure()
                if chart_type == "Line":
                    fig.add_trace(go.Scatter(x=df_x[x_col], y=df_y[y_col], mode='lines', name=f"{x_file} vs {y_file}"))
                elif chart_type == "Bar":
                    fig.add_trace(go.Bar(x=df_x[x_col], y=df_y[y_col], name=f"{x_file} vs {y_file}"))
                else:
                    fig.add_trace(go.Scatter(x=df_x[x_col], y=df_y[y_col], mode='markers', name=f"{x_file} vs {y_file}"))

                fig.update_layout(title=f"{chart_type} Chart", xaxis_title=x_col, yaxis_title=y_col)
                st.plotly_chart(fig, use_container_width=True)

    # --- Navigation ---
    if st.button("Next: Report & Export →", type="primary"):
        st.session_state.page = "Report & Export"
        st.rerun()

# -------------------------
# --- Page: Report & Export
# -------------------------
elif st.session_state.page == "Report & Export":
    if not require_login_ui(require_login=False):
        st.stop()

    st.title("📋 Report & Export")
    if not st.session_state.files:
        st.warning("Upload files first in Data Upload.")
    else:
        active = st.session_state.files[st.session_state.active_file_idx]
        df = active.get("df")
        if df is not None:
            st.subheader(f"Report for: {active['name']}")
            
            # Generate basic report
            st.write("Dataset Overview:")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Column details
            st.write("Column Details:")
            col_info = []
            for col in df.columns:
                col_info.append({
                    "Column": col,
                    "Type": str(df[col].dtype),
                    "Non-Null": df[col].count(),
                    "Null": df[col].isnull().sum(),
                    "Unique": df[col].nunique()
                })
            st.dataframe(pd.DataFrame(col_info))
            
            # Export options
            st.subheader("Export Options")
            
            export_format = st.selectbox("Choose export format:", ["CSV", "Excel", "JSON"])
            
            if st.button("Generate Export File"):
                if export_format == "CSV":
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{active['name']}_processed.csv",
                        mime="text/csv"
                    )
                elif export_format == "Excel":
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Data')
                    st.download_button(
                        label="Download Excel",
                        data=buffer.getvalue(),
                        file_name=f"{active['name']}_processed.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                elif export_format == "JSON":
                    json_str = df.to_json(orient='records')
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"{active['name']}_processed.json",
                        mime="application/json"
                    )
                    
        else:
            st.info("No DataFrame loaded for active file.")

    if st.button("Next: Hypothesis Summary →", type="primary"):
        st.session_state.page = "Hypothesis Summary"
        st.rerun()

# -------------------------
# --- Page: Hypothesis Summary
# -------------------------
elif st.session_state.page == "Hypothesis Summary":
    if not require_login_ui(require_login=False):
        st.stop()

    st.title("🔍 Hypothesis Summary")
    if not st.session_state.files:
        st.warning("Upload files first in Data Upload.")
    else:
        st.subheader("Data Analysis Summary")
        
        for i, file_entry in enumerate(st.session_state.files):
            df = file_entry.get("df")
            name = file_entry["name"]
            
            with st.expander(f"Analysis for: {name}", expanded=(i == st.session_state.active_file_idx)):
                if df is not None:
                    hypothesis_text = _generate_hypotheses_for_df(df, name)
                    st.markdown(hypothesis_text)
                    
                    # Additional insights
                    st.write("Quick Insights:")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.write(f"- Most variable numeric column: {df[numeric_cols].std().idxmax()}")
                        st.write(f"- Least variable numeric column: {df[numeric_cols].std().idxmin()}")
                    
                    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                    if len(categorical_cols) > 0:
                        most_diverse = df[categorical_cols].nunique().idxmax()
                        st.write(f"- Most diverse categorical column: {most_diverse} ({df[most_diverse].nunique()} unique values)")
                        
                else:
                    st.info(f"No data loaded for {name}")
        
        # Overall project summary
        st.subheader("📊 Project Summary")
        total_files = len(st.session_state.files)
        total_rows = sum(len(f.get("df", [])) for f in st.session_state.files if f.get("df") is not None)
        
        st.write(f"Total files processed: {total_files}")
        st.write(f"Total rows across all files: {total_rows}")
        st.write(f"User role: {st.session_state.user_role}")
        
        if st.button("🏠 Back to Data Upload", type="primary"):
            st.session_state.page = "Data Upload"
            st.rerun()