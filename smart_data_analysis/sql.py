import streamlit as st
import pandas as pd
import mysql.connector
import plotly.express as px

st.set_page_config(page_title="SQL Data Dashboard", layout="wide")

st.title("🗄 MySQL Data Analysis Dashboard")

# -------------------------
# --- 1️⃣ DATABASE CONNECTION
# -------------------------
with st.sidebar:
    st.header("🔌 Connect to MySQL Database")
    host = st.text_input("Host", "sql12.freesqldatabase.com")
    port = st.number_input("Port", 3306)
    database = st.text_input("Database Name")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Connect"):
        try:
            conn = mysql.connector.connect(
                host=host,
                port=int(port),
                user=username,
                password=password,
                database=database
            )
            st.session_state.conn = conn
            st.sidebar.success("✅ Connected successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Connection failed: {e}")

# -------------------------
# --- 2️⃣ LOAD TABLE
# -------------------------
if "conn" in st.session_state:
    conn = st.session_state.conn
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES;")
    tables = [t[0] for t in cursor.fetchall()]
    cursor.close()

    if not tables:
        st.warning("No tables found in this database.")
        st.stop()

    selected_table = st.selectbox("Select Table", tables)
    if selected_table:
        df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
        st.session_state.df = df
        st.session_state.table = selected_table
        st.success(f"Loaded table: {selected_table}")

# -------------------------
# --- 3️⃣ TOP NAV BUTTONS
# -------------------------
if "df" in st.session_state:
    selected_tab = st.radio(
        "🔹 Navigate Sections:",
        ["📜 Raw Data", "🧹 Cleaning", "📈 Visualization", "📋 Report", "🔍 Hypothesis"],
        horizontal=True
    )

    df = st.session_state.df
    table = st.session_state.table

    # -------------------------
    # --- RAW DATA
    # -------------------------
    if selected_tab == "📜 Raw Data":
        st.header(f"📜 Raw Data - {table}")
        st.dataframe(df.head(100))

        with st.expander("🔍 Dataset Info"):
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")
            st.write(f"Missing Values: {df.isnull().sum().sum()}")
            st.write("Data Types:")
            st.write(df.dtypes)

    # -------------------------
    # --- CLEANING
    # -------------------------
    elif selected_tab == "🧹 Cleaning":
        st.header("🧹 Data Cleaning Tools")

        if st.button("Remove Duplicates"):
            before = len(df)
            df = df.drop_duplicates()
            after = len(df)
            st.session_state.df = df
            st.success(f"Removed {before - after} duplicate rows.")

        if st.button("Fill Missing Values"):
            for col in df.columns:
                if df[col].dtype in ["float64", "int64"]:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
            st.session_state.df = df
            st.success("Missing values filled successfully.")

        st.dataframe(df.head(50))

    # -------------------------
    # --- VISUALIZATION
    # -------------------------
    elif selected_tab == "📈 Visualization":
        st.header("📈 Data Visualization")

        cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

        chart_type = st.selectbox("Select Chart Type", ["Histogram", "Box Plot", "Scatter", "Bar"])
        x_col = st.selectbox("Select X-axis", cols)
        y_col = st.selectbox("Select Y-axis (optional)", [None] + num_cols)

        if chart_type == "Histogram":
            fig = px.histogram(df, x=x_col)
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col)
        elif chart_type == "Scatter" and y_col:
            fig = px.scatter(df, x=x_col, y=y_col)
        elif chart_type == "Bar":
            fig = px.bar(df, x=x_col, y=y_col if y_col else None)
        else:
            fig = None

        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # --- REPORT
    # -------------------------
    elif selected_tab == "📋 Report":
        st.header("📋 Summary Report & Export")

        st.download_button(
            label="📥 Download Cleaned Data (CSV)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"{table}_cleaned.csv",
            mime="text/csv"
        )

        st.subheader("📊 Summary Statistics")
        st.dataframe(df.describe(include="all").T)

    # -------------------------
    # --- HYPOTHESIS
    # -------------------------
    elif selected_tab == "🔍 Hypothesis":
        st.header("🔍 Hypothesis Summary")

        st.markdown(f"""
        ### Insights for {table}
        - Check numeric columns for potential correlations.
        - Handle categorical data carefully — encode before ML use.
        - Investigate outliers — they may distort trends.
        - Missing values can bias results — ensure proper treatment.
        - Feature scaling might be needed for algorithms sensitive to magnitude.
        """)

else:
    st.info("👈 Connect to a database and load a table to begin.")


# -------------------------
# --- 4️⃣ DISCONNECT BUTTON
# -------------------------
if "conn" in st.session_state:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔴 Disconnect"):
        try:
            st.session_state.conn.close()
        except Exception:
            pass
        st.session_state.clear()
        st.sidebar.success("🔌 Disconnected successfully!")
        st.rerun()

    