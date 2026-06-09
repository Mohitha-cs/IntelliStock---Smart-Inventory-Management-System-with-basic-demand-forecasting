
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import sqlite3
import uuid
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import random

# ══════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="IntelliStock — Inventory Management",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "IntelliStock v1.0 | Smart Inventory Management System"},
)

DB_PATH = "stockwise.db"

# ══════════════════════════════════════════════════════════
#  AUTHENTICATION  (DB-backed — accounts persist on restart)
#  Roles:  "admin"  → full read + write access
# ══════════════════════════════════════════════════════════
import hashlib


def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def init_users_table():
    """Create the users table and seed default accounts if empty."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username   TEXT PRIMARY KEY,
                password   TEXT NOT NULL,
                full_name  TEXT NOT NULL,
                role       TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
        # Seed default accounts only if table is empty
        count = pd.read_sql_query("SELECT COUNT(*) AS n FROM users", conn).iloc[0]["n"]
        if count == 0:
            defaults = [
                ("admin",   hash_pw("admin123"),   "Administrator", "admin"),
                ("manager", hash_pw("manager123"), "Store Manager", "admin"),
            ]
            conn.executemany(
                "INSERT INTO users(username,password,full_name,role) VALUES(?,?,?,?)",
                defaults
            )
            conn.commit()


def check_login(username: str, password: str):
    """Return user row as dict if credentials match, else None."""
    row = qdf("SELECT * FROM users WHERE username=?", (username.strip().lower(),))
    if len(row) and row.iloc[0]["password"] == hash_pw(password):
        return row.iloc[0].to_dict()
    return None


def username_exists(username: str) -> bool:
    return len(qdf("SELECT 1 FROM users WHERE username=?", (username.strip().lower(),))) > 0


def create_user(username: str, full_name: str, password: str, role: str):
    run("INSERT INTO users(username,password,full_name,role) VALUES(?,?,?,?)",
        (username.strip().lower(), hash_pw(password), full_name.strip(), role))


def is_admin() -> bool:
    return st.session_state.get("role") == "admin"


def render_login():
    """Render the login / sign-up screen."""
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        # Brand header
        st.markdown("""
        <div style="margin-top:60px;text-align:center;margin-bottom:28px;">
          <div style="font-size:34px;font-weight:800;letter-spacing:-1px;
               background:linear-gradient(90deg,#3b64dc,#7c3aed);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            IntelliStock
          </div>
          <div style="font-size:12px;color:#64748b;margin-top:6px;
               text-transform:uppercase;letter-spacing:1.8px;">
            Smart Inventory Management
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Toggle between Sign In and Sign Up
        tab_in, tab_up = st.tabs(["  Sign In  ", "  Create Account  "])

        # ── SIGN IN ──────────────────────────────────
        with tab_in:
            username = st.text_input("Username", placeholder="Enter username", key="li_user")
            password = st.text_input("Password", type="password",
                                     placeholder="Enter password", key="li_pw")

            if st.button("Sign In", type="primary", use_container_width=True, key="li_btn"):
                user = check_login(username, password)
                if user:
                    st.session_state["logged_in"] = True
                    st.session_state["username"]  = user["username"]
                    st.session_state["name"]      = user["full_name"]
                    st.session_state["role"]      = user["role"]
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

        # ── SIGN UP ──────────────────────────────────
        with tab_up:

            new_name = st.text_input("Full Name",        placeholder="e.g. Jane Smith",    key="su_name")
            new_user = st.text_input("Choose Username",  placeholder="lowercase, no spaces", key="su_user")
            new_pw   = st.text_input("Password",         type="password",
                                     placeholder="Min 6 characters",   key="su_pw")
            new_pw2  = st.text_input("Confirm Password", type="password",
                                     placeholder="Repeat password",     key="su_pw2")
            new_role = st.selectbox("Account role","admin",
                                    help="admin = full access",
                                    key="su_role")

            if st.button("Create Account", type="primary", use_container_width=True, key="su_btn"):
                u = new_user.strip().lower()
                if not new_name.strip():
                    st.error("Please enter your full name.")
                elif len(u) < 3:
                    st.error("Username must be at least 3 characters.")
                elif " " in u:
                    st.error("Username cannot contain spaces.")
                elif len(new_pw) < 6:
                    st.error("Password must be at least 6 characters.")
                elif new_pw != new_pw2:
                    st.error("Passwords do not match.")
                elif username_exists(u):
                    st.error(f"Username '{u}' is already taken.")
                else:
                    create_user(u, new_name, new_pw, new_role)
                    st.success(f"Account created! You can now sign in as '{u}'.")

            st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  CSS / THEME
# ══════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family:'Inter',sans-serif !important; }
    #MainMenu, footer, header { visibility:hidden; }
    [data-testid="stDecoration"] { display:none; }

    .main { background:#f0f4f8; }
    .block-container { padding:28px 36px 48px 36px !important; max-width:1440px !important; }

    [data-testid="stSidebar"] {
        background:linear-gradient(175deg,#0c1220 0%,#161f33 60%,#1a2744 100%) !important;
        border-right:1px solid rgba(99,132,199,0.18) !important;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top:0 !important; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span { color:#8da0c4 !important; font-size:13px !important; }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] { gap:2px !important; }
    [data-testid="stSidebar"] .stRadio label {
        border-radius:8px !important; padding:9px 14px !important;
        transition:all 0.18s !important; width:100% !important;
        margin:1px 0 !important; color:#8da0c4 !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background:rgba(99,132,199,0.12) !important; color:#c8d8f0 !important;
    }

    [data-testid="stMetric"] {
        background:white !important; border-radius:14px !important;
        padding:18px 20px 16px !important; border:1px solid #e2e8f0 !important;
        box-shadow:0 2px 8px rgba(15,23,42,0.06) !important;
        border-top:3px solid !important;
        border-image:linear-gradient(90deg,#3b64dc,#7c3aed) 1 !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size:11px !important; font-weight:600 !important;
        text-transform:uppercase !important; letter-spacing:0.8px !important;
        color:#64748b !important;
    }
    [data-testid="stMetricValue"] { font-size:26px !important; font-weight:700 !important; color:#0f172a !important; }
    [data-testid="stMetricDelta"] { font-size:12px !important; }

    .stButton > button {
        border-radius:9px !important; font-weight:500 !important;
        font-size:13px !important; transition:all 0.15s !important;
        border:1px solid #e2e8f0 !important;
    }
    .stButton > button[kind="primary"] {
        background:linear-gradient(135deg,#2563eb,#4f46e5) !important;
        color:white !important; border:none !important;
        box-shadow:0 2px 8px rgba(37,99,235,0.30) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background:linear-gradient(135deg,#1d4ed8,#4338ca) !important;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stTextArea"] textarea {
        border-radius:9px !important; font-size:13px !important;
        border:1px solid #cbd5e1 !important; background:#fff !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background:#e8edf4 !important; border-radius:10px !important;
        gap:4px !important; padding:4px !important; border:none !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius:8px !important; padding:8px 18px !important;
        font-size:13px !important; font-weight:500 !important;
        color:#64748b !important; background:transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background:white !important; color:#0f172a !important;
        box-shadow:0 2px 6px rgba(0,0,0,0.10) !important;
    }

    [data-testid="stDataFrame"] {
        border-radius:12px !important; overflow:hidden !important;
        border:1px solid #e2e8f0 !important;
        box-shadow:0 1px 4px rgba(0,0,0,0.04) !important;
    }

    .page-header { margin-bottom:22px; padding-bottom:16px; border-bottom:1px solid #e2e8f0; }
    .page-header h2 {
        font-size:21px; font-weight:700; margin:0;
        background:linear-gradient(90deg,#1e3a8a,#4f46e5);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    }
    .page-header p { font-size:13px; color:#64748b; margin:5px 0 0 0; }

    .sw-card {
        background:white; border-radius:14px; padding:22px 24px;
        border:1px solid #e2e8f0; box-shadow:0 2px 8px rgba(15,23,42,0.06);
        margin-bottom:16px;
    }
    .section-title {
        font-size:13px; font-weight:600; color:#1e293b;
        padding-left:10px; border-left:3px solid #3b64dc;
        margin-bottom:12px; margin-top:4px;
    }

    .alert-critical {
        background:linear-gradient(90deg,#fef2f2,#fff5f5);
        border-left:4px solid #ef4444; border-radius:10px;
        padding:12px 16px; margin:5px 0;
    }
    .alert-warning {
        background:linear-gradient(90deg,#fffbeb,#fffcf0);
        border-left:4px solid #f59e0b; border-radius:10px;
        padding:12px 16px; margin:5px 0;
    }
    .alert-ok {
        background:linear-gradient(90deg,#f0fdf4,#f5fef7);
        border-left:4px solid #10b981; border-radius:10px;
        padding:12px 16px; margin:5px 0;
    }
    .alert-info {
        background:linear-gradient(90deg,#eff6ff,#f0f7ff);
        border-left:4px solid #3b82f6; border-radius:10px;
        padding:12px 16px; margin:5px 0;
    }
    .alert-title { font-weight:600; font-size:13px; color:#1e293b; }
    .alert-body  { font-size:12px; color:#475569; margin-top:3px; }

    .badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600; }
    .badge-red    { background:#fef2f2; color:#dc2626; }
    .badge-amber  { background:#fffbeb; color:#d97706; }
    .badge-green  { background:#f0fdf4; color:#16a34a; }
    .badge-blue   { background:#eff6ff; color:#2563eb; }
    .badge-purple { background:#faf5ff; color:#7c3aed; }

    .rule-card {
        background:white; border-radius:12px; border:1px solid #e2e8f0;
        padding:14px 18px; margin-bottom:10px;
        box-shadow:0 1px 4px rgba(0,0,0,0.04);
        display:flex; align-items:center; gap:16px; flex-wrap:wrap;
    }
    .rule-lhs {
        background:linear-gradient(135deg,#eff6ff,#e8f0fe);
        border-radius:8px; padding:8px 12px; font-size:12px;
        font-weight:600; color:#1e40af; min-width:140px; text-align:center;
    }
    .rule-arrow { font-size:22px; color:#94a3b8; }
    .rule-rhs {
        background:linear-gradient(135deg,#f0fdf4,#e8fef0);
        border-radius:8px; padding:8px 12px; font-size:12px;
        font-weight:600; color:#166534; min-width:140px; text-align:center;
    }
    .rule-stats { margin-left:auto; display:flex; gap:20px; }
    .rule-stat { text-align:center; }
    .rule-stat-val { font-size:15px; font-weight:700; color:#0f172a; }
    .rule-stat-lbl { font-size:10px; color:#94a3b8; text-transform:uppercase; letter-spacing:0.5px; }

    hr { border-color:#e2e8f0 !important; margin:20px 0 !important; }
    [data-testid="stExpander"] {
        border-radius:10px !important; border:1px solid #e2e8f0 !important;
        background:white !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL UNIQUE,
            category    TEXT    NOT NULL DEFAULT 'General',
            brand       TEXT    DEFAULT '',
            description TEXT    DEFAULT '',
            unit        TEXT    DEFAULT 'units',
            sell_price  REAL    NOT NULL DEFAULT 0.0,
            cost_price  REAL    NOT NULL DEFAULT 0.0,
            reorder_pt  INTEGER NOT NULL DEFAULT 10,
            max_stock   INTEGER NOT NULL DEFAULT 100,
            active      INTEGER NOT NULL DEFAULT 1,
            created_at  TEXT    DEFAULT (date('now'))
        );
        CREATE TABLE IF NOT EXISTS inventory (
            product_id  INTEGER PRIMARY KEY,
            quantity    INTEGER NOT NULL DEFAULT 0,
            updated_at  TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY(product_id) REFERENCES products(id)
        );
        CREATE TABLE IF NOT EXISTS sales (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT    NOT NULL,
            product_id     INTEGER NOT NULL,
            quantity       INTEGER NOT NULL,
            unit_price     REAL    NOT NULL,
            total          REAL    NOT NULL,
            sale_date      TEXT    NOT NULL,
            customer       TEXT    DEFAULT '',
            notes          TEXT    DEFAULT '',
            created_at     TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS stock_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id  INTEGER NOT NULL,
            type        TEXT    NOT NULL DEFAULT 'restock',
            qty_change  INTEGER NOT NULL,
            qty_after   INTEGER NOT NULL DEFAULT 0,
            reason      TEXT    DEFAULT '',
            logged_at   TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY(product_id) REFERENCES products(id) ON DELETE CASCADE
        );
        """)


init_db()
init_users_table()  # ensure users table + default accounts exist


def qdf(query, params=()):
    with get_db() as conn:
        return pd.read_sql_query(query, conn, params=list(params))


def run(query, params=()):
    with get_db() as conn:
        cur = conn.execute(query, list(params))
        conn.commit()
        return cur.lastrowid


def runmany(query, data):
    with get_db() as conn:
        conn.executemany(query, data)
        conn.commit()


# ── Product helpers ──────────────────────────────────────
def get_products(active_only=True):
    cond = "WHERE p.active=1" if active_only else ""
    return qdf(f"""
        SELECT p.*, COALESCE(i.quantity,0) AS stock, COALESCE(i.updated_at,'—') AS last_updated
        FROM products p LEFT JOIN inventory i ON p.id=i.product_id
        {cond} ORDER BY p.category, p.name
    """)


def get_product_map():
    df = get_products()
    return dict(zip(df["name"], df["id"]))


def add_product(name, category, brand, description, unit, sell_price, cost_price, reorder_pt, max_stock):
    pid = run("""INSERT INTO products(name,category,brand,description,unit,sell_price,cost_price,reorder_pt,max_stock)
                 VALUES(?,?,?,?,?,?,?,?,?)""",
              (name, category, brand, description, unit, sell_price, cost_price, reorder_pt, max_stock))
    run("INSERT OR IGNORE INTO inventory(product_id,quantity) VALUES(?,0)", (pid,))
    return pid


def update_product(pid, name, category, description, unit, sell_price, cost_price, reorder_pt, max_stock):
    run("""UPDATE products SET name=?,category=?,description=?,unit=?,sell_price=?,
           cost_price=?,reorder_pt=?,max_stock=? WHERE id=?""",
        (name, category, description, unit, sell_price, cost_price, reorder_pt, max_stock, pid))


def delete_product(pid):
    run("UPDATE products SET active=0 WHERE id=?", (pid,))


# ── Inventory helpers ────────────────────────────────────
def get_inventory():
    return qdf("""
        SELECT p.id, p.name, p.category, p.brand, p.unit,
               p.sell_price, p.cost_price, p.reorder_pt, p.max_stock,
               COALESCE(i.quantity,0) AS stock,
               COALESCE(i.updated_at,'—') AS updated_at,
               CASE
                 WHEN COALESCE(i.quantity,0) = 0               THEN 'Out of Stock'
                 WHEN COALESCE(i.quantity,0) < p.reorder_pt    THEN 'Low Stock'
                 WHEN COALESCE(i.quantity,0) >= p.max_stock*0.8 THEN 'Well Stocked'
                 ELSE 'Adequate'
               END AS status
        FROM products p LEFT JOIN inventory i ON p.id=i.product_id
        WHERE p.active=1 ORDER BY stock ASC
    """)


def set_stock(product_id, new_qty, move_type="restock", reason=""):
    old = qdf("SELECT COALESCE(quantity,0) AS q FROM inventory WHERE product_id=?", (product_id,))
    old_qty = int(old["q"].iloc[0]) if len(old) else 0
    run("INSERT OR REPLACE INTO inventory(product_id,quantity,updated_at) VALUES(?,?,datetime('now'))",
        (product_id, new_qty))
    run("INSERT INTO stock_log(product_id,type,qty_change,qty_after,reason) VALUES(?,?,?,?,?)",
        (product_id, move_type, new_qty - old_qty, new_qty, reason))


def adjust_stock(product_id, delta, move_type="adjustment", reason=""):
    old = qdf("SELECT COALESCE(quantity,0) AS q FROM inventory WHERE product_id=?", (product_id,))
    old_qty = int(old["q"].iloc[0]) if len(old) else 0
    set_stock(product_id, max(0, old_qty + delta), move_type, reason)


# ── Sales helpers ────────────────────────────────────────
def record_sale(product_id, quantity, unit_price, sale_date, customer="", notes="", txn_id=None):
    txn = txn_id or str(uuid.uuid4())
    run("""INSERT INTO sales(transaction_id,product_id,quantity,unit_price,total,sale_date,customer,notes)
           VALUES(?,?,?,?,?,?,?,?)""",
        (txn, product_id, quantity, unit_price, quantity * unit_price, str(sale_date), customer, notes))
    adjust_stock(product_id, -quantity, "sale", f"Sale {txn[:8]}")


def get_sales(days=None):
    clause = f"WHERE s.sale_date >= date('now','-{days} days')" if days else ""
    return qdf(f"""
        SELECT s.id, s.transaction_id, p.name AS product, p.category,
               s.quantity, s.unit_price, s.total, s.sale_date, s.customer, s.notes
        FROM sales s JOIN products p ON s.product_id=p.id
        {clause} ORDER BY s.sale_date DESC, s.created_at DESC
    """)


def get_product_daily_sales(product_id):
    return qdf("""
        SELECT sale_date, SUM(quantity) AS quantity, SUM(total) AS revenue
        FROM sales WHERE product_id=?
        GROUP BY sale_date ORDER BY sale_date
    """, (product_id,))


# ══════════════════════════════════════════════════════════
#  WALMART-STYLE DATASET
# ══════════════════════════════════════════════════════════
WALMART_PRODUCTS = [
    # (name, category, brand, description, unit, sell_price, cost_price, reorder_pt, max_stock)
    # Baby & Toddler
    ("Pampers Baby-Dry Diapers Sz3 80ct",    "Baby & Toddler","Pampers",  "Disposable diapers size 3",             "pack",  28.97,14.50,20,120),
    ("Huggies Little Snugglers Sz2 84ct",    "Baby & Toddler","Huggies",  "Ultra-soft newborn diapers size 2",     "pack",  27.47,13.80,20,120),
    ("Similac Advance Baby Formula 30oz",    "Baby & Toddler","Similac",  "Iron-fortified powder infant formula",  "can",   26.98,13.40,15, 80),
    ("WaterWipes Sensitive Baby Wipes 240ct","Baby & Toddler","WaterWipes","99.9% water purity baby wipes",        "pack",  14.97, 7.20,25,150),
    # Beverages
    ("Budweiser Beer 12-Pack Cans",          "Beverages","Budweiser","American lager 12x12fl oz",                 "pack",  14.97, 7.80,30,200),
    ("Coca-Cola Classic 24-Pack Cans",       "Beverages","Coca-Cola","Original taste 24x12fl oz",                "pack",   9.98, 5.10,40,250),
    ("Minute Maid Orange Juice 89oz",        "Beverages","Minute Maid","100% pure squeezed OJ",                  "bottle", 4.97, 2.20,35,200),
    ("Folgers Classic Roast Coffee 43.5oz",  "Beverages","Folgers",  "Medium roast ground coffee",               "can",   11.98, 5.90,20,120),
    ("Gatorade Thirst Quencher 12-Pack",     "Beverages","Gatorade", "Assorted flavour sports drinks",           "pack",  13.97, 6.80,25,150),
    ("Monster Energy Drink 24-Pack",         "Beverages","Monster",  "Original green energy drinks 16fl oz",     "pack",  39.97,20.50,15,100),
    # Grocery
    ("Wonder Classic White Bread 20oz",      "Grocery","Wonder",    "Soft white sandwich bread",                 "loaf",   3.18, 1.20,50,400),
    ("Land O Lakes Unsalted Butter 1lb",     "Grocery","Land O Lakes","Grade A sweet cream butter",              "pack",   5.48, 2.60,40,300),
    ("Great Value Large Eggs 18ct",          "Grocery","Great Value","Grade A large white eggs",                 "pack",   3.68, 1.70,50,400),
    ("Great Value Whole Milk 1 Gallon",      "Grocery","Great Value","Vitamin D whole milk",                     "jug",    3.48, 1.60,50,400),
    ("Kraft American Cheese Slices 16oz",    "Grocery","Kraft",     "Individually wrapped singles",              "pack",   4.98, 2.30,40,280),
    ("Lay's Classic Potato Chips 13oz",      "Grocery","Lay's",     "Classic salted potato chips",               "bag",    4.98, 2.10,40,300),
    ("Oreo Original Cookies 14.3oz",         "Grocery","Oreo",      "Classic sandwich cookies",                  "pack",   3.98, 1.70,40,280),
    ("Nabisco Ritz Crackers 13.7oz",         "Grocery","Nabisco",   "Original butter crackers",                  "pack",   3.98, 1.80,35,250),
    # Electronics
    ("Duracell Coppertop AA Batteries 20ct", "Electronics","Duracell","Long-lasting alkaline AA",               "pack",  12.97, 5.90,30,200),
    ("Anker USB-C to USB-C Cable 6ft",       "Electronics","Anker",   "60W braided fast-charge cable",          "each",   8.99, 3.40,20,150),
    ("JLab Go Air Pop True Wireless Earbuds","Electronics","JLab",    "Bluetooth 5.1 TWS earbuds",              "each",  19.88, 8.20,15,100),
    ("Onn 18W USB-C Fast Charger",           "Electronics","Onn",     "Wall adapter USB-C PD port",             "each",  12.88, 4.90,20,120),
    ("Duracell Coppertop AAA Batteries 16ct","Electronics","Duracell","Long-lasting alkaline AAA",              "pack",  10.97, 4.80,25,180),
    # Personal Care
    ("Head & Shoulders 2-in-1 Shampoo 23.7oz","Personal Care","Head & Shoulders","Anti-dandruff shampoo+cond","bottle", 7.97, 3.20,30,200),
    ("TRESemme Moisture Rich Conditioner 28oz","Personal Care","TRESemme",        "Salon-quality conditioner",  "bottle", 4.97, 1.90,30,200),
    ("Colgate Total Toothpaste 4-Pack",      "Personal Care","Colgate",  "Whitening cavity protection 4x5.1oz","pack",   8.97, 3.80,25,160),
    ("Old Spice Original Deodorant 3-Pack",  "Personal Care","Old Spice","48h odour protection 3x3oz",          "pack",   9.97, 4.30,25,150),
    ("BIC Silky Touch Razors 15ct",          "Personal Care","BIC",     "Disposable razors",                    "pack",   7.97, 3.10,20,140),
    # Home & Cleaning
    ("Bounty Select-A-Size Paper Towels 8pk","Home & Cleaning","Bounty",  "Double-plus select-a-size rolls",  "pack",  14.97, 6.80,30,180),
    ("Gain Original Detergent 154 fl oz",    "Home & Cleaning","Gain",    "HE-compatible liquid laundry soap","bottle",13.97, 6.20,25,150),
    ("Dawn Ultra Dish Soap 70 fl oz",        "Home & Cleaning","Dawn",    "Original grease-cutting soap",     "bottle", 7.97, 3.30,30,200),
    ("Downy Ultra Fabric Softener 103oz",    "Home & Cleaning","Downy",   "Fresh scent fabric conditioner",   "bottle", 9.97, 4.20,25,160),
    ("Lysol Disinfecting Wipes 3-Pack",      "Home & Cleaning","Lysol",   "Lemon & lime multi-surface wipes", "pack",  10.97, 4.90,25,160),
    # Sports & Fitness
    ("Optimum Nutrition Whey Protein 5lb",   "Sports & Fitness","ON",       "Gold standard whey protein",     "tub",   54.97,27.00,10, 60),
    ("Pure Protein Bars Variety 18ct",       "Sports & Fitness","Pure Protein","High-protein bars 50g",       "pack",  24.97,12.00,15, 90),
    # Auto
    ("Pennzoil Platinum 5W-30 Motor Oil 5qt","Auto","Pennzoil","Full synthetic motor oil",                     "jug",   22.97,11.50,20,120),
    ("Rain-X Washer Fluid -20F 128oz",       "Auto","Rain-X",  "Winter windshield washer fluid",               "jug",    4.97, 1.80,30,200),
    # Clothing
    ("Hanes Men's ComfortSoft T-Shirts 5pk", "Clothing","Hanes",           "100% cotton crewneck tees",       "pack",  14.97, 6.50,20,150),
    ("Fruit of Loom Women's Socks 10-Pack",  "Clothing","Fruit of Loom",   "No-show ankle socks white",       "pack",   8.97, 3.60,20,150),
    # Pharmacy
    ("Tylenol Extra Strength 500mg 100ct",   "Pharmacy","Tylenol","Acetaminophen pain reliever",               "bottle",11.97, 5.20,20,130),
    ("Advil Ibuprofen 200mg 100ct",          "Pharmacy","Advil",  "NSAID pain reliever/fever reducer",         "bottle", 9.97, 4.30,20,130),
    ("Band-Aid Flexible Fabric Bandages 100ct","Pharmacy","Band-Aid","Assorted flexible bandages",             "box",    6.97, 2.70,20,140),
]

INITIAL_STOCK = [
    65,70,35,90,
    180,220,160,80,110,55,
    380,280,360,370,260,270,240,220,
    155,95,60,88,130,
    165,155,120,130,110,
    140,120,170,130,140,
    28,52,
    95,165,
    110,110,
    100,95,105,
]

DEMAND_PARAMS = [
    (3.5,1.5,0.10),(3.2,1.4,0.08),(1.8,0.8,0.06),(5.0,2.0,0.07),
    (8.5,3.0,0.05),(12.0,4.0,0.04),(9.0,3.5,0.03),(4.5,1.8,0.05),
    (6.0,2.5,0.06),(2.5,1.0,0.09),
    (22.0,7.0,0.02),(15.0,5.0,0.02),(20.0,6.5,0.03),(22.0,7.0,0.02),
    (13.0,4.5,0.02),(13.5,5.0,0.03),(12.0,4.5,0.03),(11.5,4.0,0.02),
    (7.0,2.5,0.04),(4.5,2.0,0.06),(2.8,1.2,0.08),(4.0,1.8,0.05),(6.0,2.2,0.03),
    (8.0,3.0,0.03),(7.5,2.8,0.03),(5.5,2.2,0.02),(6.0,2.5,0.02),(4.5,1.8,0.03),
    (7.0,2.8,0.03),(6.0,2.5,0.03),(8.5,3.2,0.03),(5.5,2.2,0.03),(6.5,2.8,0.03),
    (1.5,0.7,0.10),(2.8,1.2,0.07),
    (4.5,1.8,0.04),(8.5,3.0,0.03),
    (5.5,2.2,0.04),(5.0,2.0,0.04),
    (4.5,1.8,0.04),(3.8,1.5,0.04),(5.5,2.0,0.03),
]

# (anchor_product_index, associated_product_index, probability)
BASKET_RULES = [
    (0, 4,  0.42), (1, 4,  0.38),  # Diapers -> Beer (classic!)
    (0, 3,  0.55), (1, 3,  0.52),  # Diapers -> Baby Wipes
    (2, 3,  0.48),                  # Formula -> Baby Wipes
    (10,11, 0.58), (10,12, 0.50),  # Bread -> Butter, Eggs
    (12,13, 0.55), (12,10, 0.48),  # Eggs -> Milk, Bread
    (13,11, 0.45),                  # Milk -> Butter
    (15,5,  0.50), (16,5,  0.42),  # Chips/Cookies -> Cola
    (17,5,  0.38),                  # Crackers -> Cola
    (23,24, 0.68), (24,23, 0.62),  # Shampoo <-> Conditioner
    (19,20, 0.45), (20,19, 0.40),  # USB-C Cable <-> Earbuds
    (19,21, 0.42),                  # Cable -> Charger
    (18,22, 0.48),                  # AA -> AAA Batteries
    (33,34, 0.50), (9, 34, 0.40),  # Protein <-> Energy Drink
    (29,31, 0.55), (31,29, 0.52),  # Detergent <-> Fabric Softener
    (28,30, 0.45), (30,32, 0.42),  # Paper Towels -> Dish Soap -> Lysol
    (25,26, 0.44),                  # Toothpaste -> Deodorant
    (35,36, 0.48),                  # Motor Oil -> Washer Fluid
    (39,40, 0.42),                  # Tylenol <-> Advil
    (6, 7,  0.40),                  # OJ -> Coffee
]


def seed_walmart_data():
    """Seed Walmart-style dataset on first run — runs silently at startup."""
    if qdf("SELECT COUNT(*) AS n FROM products").iloc[0]["n"] > 0:
        return

    random.seed(42)
    np.random.seed(42)

    pids = [add_product(*p) for p in WALMART_PRODUCTS]
    for pid, stock in zip(pids, INITIAL_STOCK):
        set_stock(pid, stock, "initial_stock", "Initial Walmart-style stock")

    today       = date.today()
    idx_to_pid  = {i: pids[i] for i in range(len(pids))}
    pid_price   = {pids[i]: float(WALMART_PRODUCTS[i][5]) for i in range(len(pids))}
    assoc_map   = {}
    for anchor_i, assoc_i, prob in BASKET_RULES:
        assoc_map.setdefault(anchor_i, []).append((assoc_i, prob))

    sales = []
    for d in range(90, 0, -1):
        sale_day = today - timedelta(days=d)
        for _ in range(random.randint(40, 80)):
            txn_id = str(uuid.uuid4())
            anchors = random.sample(range(len(WALMART_PRODUCTS)),
                                    random.choices([1,2,3,4], weights=[0.45,0.30,0.15,0.10])[0])
            basket = set(anchors)
            for ai in anchors:
                for assoc_i, prob in assoc_map.get(ai, []):
                    if random.random() < prob:
                        basket.add(assoc_i)
            for bi in basket:
                avg, std, _ = DEMAND_PARAMS[bi]
                qty   = max(1, int(np.random.normal(avg * 0.12, std * 0.08)))
                price = pid_price[idx_to_pid[bi]]
                sales.append((txn_id, idx_to_pid[bi], qty, price, qty * price,
                               str(sale_day), "", ""))

    runmany("""INSERT INTO sales
               (transaction_id,product_id,quantity,unit_price,total,sale_date,customer,notes)
               VALUES(?,?,?,?,?,?,?,?)""", sales)


# ══════════════════════════════════════════════════════════
#  CHART HELPERS
#  NOTE: 'legend' is intentionally NOT in CHART_LAYOUT to
#  prevent the duplicate-keyword error in update_layout().
# ══════════════════════════════════════════════════════════
CHART_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter, sans-serif", size=12, color="#374151"),
    margin=dict(t=36, b=30, l=10, r=10),
    hovermode="x unified",
)
LEGEND_STYLE = dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11))

COLOR_PRIMARY = "#3b64dc"
COLOR_SUCCESS = "#10b981"
COLOR_WARNING = "#f59e0b"
COLOR_DANGER  = "#ef4444"
COLOR_PURPLE  = "#7c3aed"
COLOR_TEAL    = "#0891b2"

CATEGORY_COLORS = {
    "Baby & Toddler":   "#ec4899",
    "Beverages":        "#3b64dc",
    "Grocery":          "#10b981",
    "Electronics":      "#7c3aed",
    "Personal Care":    "#0891b2",
    "Home & Cleaning":  "#f59e0b",
    "Sports & Fitness": "#ef4444",
    "Auto":             "#64748b",
    "Clothing":         "#d97706",
    "Pharmacy":         "#059669",
    "General":          "#94a3b8",
}


# ══════════════════════════════════════════════════════════
#  DEMAND FORECASTING
# ══════════════════════════════════════════════════════════
def run_forecast(product_id, days_ahead=30):
    raw = get_product_daily_sales(product_id)
    if len(raw) < 5:
        return None, None, None, None

    raw["sale_date"] = pd.to_datetime(raw["sale_date"])
    raw = raw.sort_values("sale_date")
    full_dates = pd.date_range(raw["sale_date"].min(), raw["sale_date"].max())
    hist = (pd.DataFrame({"date": full_dates})
              .merge(raw.rename(columns={"sale_date":"date"}), on="date", how="left")
              .fillna({"quantity": 0, "revenue": 0}))
    hist["day_num"]  = np.arange(len(hist), dtype=float)
    hist["quantity"] = hist["quantity"].astype(float)

    X = hist["day_num"].values.reshape(-1, 1)
    y = hist["quantity"].values
    model = LinearRegression()
    model.fit(X, y)
    y_hat = np.maximum(model.predict(X), 0)

    hist["predicted"] = np.round(y_hat, 2)
    avg_pos = float(np.mean(y[y > 0])) if np.any(y > 0) else 0.0
    slope   = float(model.coef_[0])

    last_t    = float(hist["day_num"].max())
    fut_t     = np.arange(last_t + 1, last_t + days_ahead + 1, dtype=float).reshape(-1, 1)
    fut_y     = np.maximum(model.predict(fut_t), 0)
    # FIX: use pd.Timedelta instead of integer arithmetic on Timestamp
    fut_dates = pd.date_range(
        hist["date"].max() + pd.Timedelta(days=1), periods=days_ahead
    )

    forecast = pd.DataFrame({"date": fut_dates, "forecasted_qty": np.round(fut_y, 2)})
    metrics  = {
        "r2":        max(0.0, round(float(r2_score(y, y_hat)), 4)),
        "mae":       round(float(mean_absolute_error(y, y_hat)), 3),
        "slope":     round(slope, 5),
        "avg_daily": round(avg_pos, 2),
        "trend":     "Increasing" if slope > 0.02 else ("Decreasing" if slope < -0.02 else "Stable"),
        "n_days":    len(hist),
        "total_sold": int(y.sum()),
    }
    return hist, forecast, metrics, avg_pos


def predict_stockout_days(product_id, current_stock):
    raw = get_product_daily_sales(product_id)
    if len(raw) < 3 or current_stock <= 0:
        return None
    raw["sale_date"] = pd.to_datetime(raw["sale_date"])
    cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=30)
    recent = raw[raw["sale_date"] >= cutoff]
    if len(recent) == 0:
        recent = raw.tail(14)
    avg_daily = recent["quantity"].sum() / 30.0
    return round(current_stock / avg_daily) if avg_daily > 0 else None


# ══════════════════════════════════════════════════════════
#  MARKET BASKET ANALYSIS
# ══════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def get_association_rules(min_support=0.01, days_back=60):
    clause = f"WHERE s.sale_date >= date('now','-{days_back} days')"
    txns = qdf(f"""
        SELECT s.transaction_id, p.name AS product
        FROM sales s JOIN products p ON s.product_id=p.id
        {clause} ORDER BY s.transaction_id
    """)
    if len(txns) < 50:
        return None, None, None

    basket = txns.groupby("transaction_id")["product"].apply(list)
    basket = basket[basket.apply(len) >= 2]
    if len(basket) < 20:
        return None, None, None

    te     = TransactionEncoder()
    matrix = pd.DataFrame(te.fit_transform(basket.tolist()), columns=te.columns_)

    try:
        freq_sets = apriori(matrix, min_support=min_support, use_colnames=True, max_len=3)
        if len(freq_sets) == 0:
            return None, None, None
        rules = association_rules(freq_sets, metric="lift", min_threshold=1.0,
                                  num_itemsets=len(freq_sets))
        rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
        return freq_sets, rules, len(basket)
    except Exception:
        return None, None, None


# ══════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════
def page_header(title, subtitle=""):
    st.markdown(f"""
    <div class="page-header">
      <h2>{title}</h2>
      {'<p>' + subtitle + '</p>' if subtitle else ''}
    </div>""", unsafe_allow_html=True)


def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
#  PAGES
# ══════════════════════════════════════════════════════════

# ── Dashboard ────────────────────────────────────────────
def page_dashboard():
    page_header("Dashboard")

    inv          = get_inventory()
    sales        = get_sales(30)
    total_val    = float((inv["stock"] * inv["sell_price"]).sum())
    low_critical = len(inv[inv["status"].isin(["Low Stock","Out of Stock"])])
    monthly_rev  = float(sales["total"].sum()) if len(sales) else 0.0
    today_rev    = float(get_sales(1)["total"].sum()) if len(get_sales(1)) else 0.0

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Products",    len(inv))
    c2.metric("Stock Value",       f"${total_val:,.0f}")
    c3.metric("Low / Critical",    low_critical,
              delta=f"{low_critical} need attention" if low_critical else None,
              delta_color="inverse")
    c4.metric("Revenue (30 days)", f"${monthly_rev:,.0f}")
    c5.metric("Today Revenue",     f"${today_rev:,.2f}")

    st.markdown("---")
    cL, cR = st.columns([3, 2])

    with cL:
        section_title("Sales trend — last 30 days")
        if len(sales):
            daily = sales.groupby("sale_date")["total"].sum().reset_index()
            daily["sale_date"] = pd.to_datetime(daily["sale_date"])
            fig = go.Figure(go.Scatter(
                x=daily["sale_date"], y=daily["total"],
                mode="lines+markers",
                line=dict(color=COLOR_PRIMARY, width=2.5),
                marker=dict(size=4),
                fill="tozeroy", fillcolor="rgba(59,100,220,0.08)",
                hovertemplate="$%{y:,.2f}<extra></extra>",
            ))
            fig.update_layout(**CHART_LAYOUT, legend=LEGEND_STYLE, height=240,
                              yaxis=dict(tickprefix="$", gridcolor="#f0f4f8"),
                              xaxis=dict(gridcolor="#f0f4f8"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sales data yet.")

    with cR:
        section_title("Units in stock by category")
        cat_s = inv.groupby("category")["stock"].sum().reset_index().sort_values("stock", ascending=False)
        fig2  = go.Figure(go.Bar(
            x=cat_s["category"], y=cat_s["stock"],
            marker_color=[CATEGORY_COLORS.get(c,"#94a3b8") for c in cat_s["category"]],
            hovertemplate="%{x}: %{y}<extra></extra>",
        ))
        fig2.update_layout(**CHART_LAYOUT, legend=LEGEND_STYLE, height=240,
                           xaxis=dict(tickfont_size=10),
                           yaxis=dict(gridcolor="#f0f4f8"))
        st.plotly_chart(fig2, use_container_width=True)

    cA, cB = st.columns(2)
    with cA:
        section_title("Top 5 products — last 30 days")
        if len(sales):
            top5 = (sales.groupby("product")
                        .agg(units=("quantity","sum"), revenue=("total","sum"))
                        .sort_values("revenue", ascending=False).head(5).reset_index())
            top5["revenue"] = top5["revenue"].apply(lambda x: f"${x:,.2f}")
            st.dataframe(top5.rename(columns={"product":"Product","units":"Units","revenue":"Revenue"}),
                         use_container_width=True, hide_index=True)
    with cB:
        section_title("Items needing attention")
        attn = inv[inv["status"].isin(["Out of Stock","Low Stock"])].head(6)
        if len(attn):
            for _, row in attn.iterrows():
                cls = "alert-critical" if row["status"] == "Out of Stock" else "alert-warning"
                st.markdown(f"""
                <div class="{cls}">
                  <div class="alert-title">{row['name']}</div>
                  <div class="alert-body">
                    {row['category']} &nbsp;·&nbsp; Stock: <b>{row['stock']} {row['unit']}</b>
                    &nbsp;·&nbsp; Reorder at: {row['reorder_pt']}
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-ok"><div class="alert-title">All products are sufficiently stocked</div></div>', unsafe_allow_html=True)


# ── Product Management ───────────────────────────────────
def page_products():
    page_header("Product Management")
    tab1, tab2 = st.tabs(["  Product Catalogue  ", "  Add New Product  "])

    CATS = ["Baby & Toddler","Beverages","Grocery","Electronics","Personal Care",
            "Home & Cleaning","Sports & Fitness","Auto","Clothing","Pharmacy","General"]

    with tab1:
        products   = get_products()
        cat_filter = st.selectbox("Filter by category", ["All"] + sorted(products["category"].unique().tolist()))
        if cat_filter != "All":
            products = products[products["category"] == cat_filter]
        st.markdown(f"**{len(products)} products**")

        for _, row in products.iterrows():
            with st.expander(f"{row['name']}  —  {row['category']}  |  Stock: {row['stock']} {row['unit']}"):
                c1,c2,c3 = st.columns(3)
                c1.markdown(f"**Brand:** {row.get('brand','—')}  \n**Sell:** ${row['sell_price']:.2f}  \n**Cost:** ${row['cost_price']:.2f}")
                c2.markdown(f"**Reorder at:** {row['reorder_pt']}  \n**Max stock:** {row['max_stock']}")
                c3.markdown(f"**Unit:** {row['unit']}  \n**Added:** {row['created_at']}")

                st.markdown("---")
                e1,e2,e3,e4,e5,e6,e7 = st.columns(7)
                new_name = e1.text_input("Name",     row["name"],  key=f"pn_{row['id']}")
                cur_cat  = row["category"] if row["category"] in CATS else "General"
                new_cat  = e2.selectbox("Category", CATS, index=CATS.index(cur_cat), key=f"pc_{row['id']}")
                new_sell = e3.number_input("Sell $", value=float(row["sell_price"]),  step=0.01, key=f"ps_{row['id']}")
                new_cost = e4.number_input("Cost $", value=float(row["cost_price"]),  step=0.01, key=f"pcp_{row['id']}")
                new_rop  = e5.number_input("Reorder",value=int(row["reorder_pt"]),    step=1,    key=f"pr_{row['id']}")
                new_max  = e6.number_input("Max",    value=int(row["max_stock"]),      step=1,    key=f"pm_{row['id']}")
                new_unit = e7.text_input("Unit",     row["unit"],                               key=f"pu_{row['id']}")

                ba, bb = st.columns([1,8])
                if ba.button("Save", key=f"save_{row['id']}", type="primary"):
                    update_product(int(row["id"]), new_name, new_cat, row["description"], new_unit, new_sell, new_cost, new_rop, new_max)
                    st.success("Saved.")
                    st.rerun()
                if bb.button("Deactivate", key=f"del_{row['id']}"):
                    delete_product(int(row["id"]))
                    st.warning("Product deactivated.")
                    st.rerun()

    with tab2:
        st.markdown("#### New Product Details")
        c1,c2 = st.columns(2)
        p_name  = c1.text_input("Product Name *")
        p_cat   = c2.selectbox("Category *", CATS)
        p_brand = st.text_input("Brand")
        p_desc  = st.text_area("Description", height=70)
        c3,c4,c5 = st.columns(3)
        p_unit  = c3.text_input("Unit", value="units")
        p_sell  = c4.number_input("Selling Price ($) *", min_value=0.0, step=0.01)
        p_cost  = c5.number_input("Cost Price ($) *",    min_value=0.0, step=0.01)
        c6,c7 = st.columns(2)
        p_rop   = c6.number_input("Reorder Point *", min_value=0, step=1, value=10)
        p_max   = c7.number_input("Max Stock *",     min_value=1, step=1, value=100)

        if st.button("Add Product", type="primary"):
            if not p_name.strip():
                st.error("Product name is required.")
            elif p_sell <= 0:
                st.error("Selling price must be > 0.")
            else:
                try:
                    add_product(p_name.strip(), p_cat, p_brand, p_desc, p_unit, p_sell, p_cost, p_rop, p_max)
                    st.success(f"'{p_name}' added.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")


# ── Inventory Monitoring ──────────────────────────────────
def page_inventory():
    page_header("Inventory Monitoring")
    tab1, tab2, tab3 = st.tabs(["  Stock Levels  ","  Restock / Adjust  ","  Movement Log  "])

    with tab1:
        inv = get_inventory()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Products",  len(inv))
        c2.metric("Well Stocked",    len(inv[inv["status"]=="Well Stocked"]))
        c3.metric("Low Stock",       len(inv[inv["status"]=="Low Stock"]),      delta_color="inverse")
        c4.metric("Out of Stock",    len(inv[inv["status"]=="Out of Stock"]),   delta_color="inverse")
        st.markdown("---")

        inv_s = inv.sort_values("stock", ascending=True).tail(20)
        bar_colors = [
            COLOR_DANGER  if s=="Out of Stock" else
            COLOR_WARNING if s=="Low Stock"    else
            COLOR_SUCCESS if s=="Well Stocked" else
            COLOR_PRIMARY
            for s in inv_s["status"]
        ]
        # FIX: use Scatter markers for reorder point instead of second Bar trace
        # to avoid the 'multiple values for legend' error that occurs with barmode+update_layout
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=inv_s["name"], x=inv_s["stock"], orientation="h",
            marker_color=bar_colors, name="Current Stock",
            text=inv_s["stock"].astype(str), textposition="outside",
            hovertemplate="%{y}: %{x} units<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            y=inv_s["name"], x=inv_s["reorder_pt"],
            mode="markers", name="Reorder Point",
            marker=dict(color=COLOR_DANGER, symbol="line-ns-open", size=14,
                        line=dict(width=2.5, color=COLOR_DANGER)),
            hovertemplate="Reorder at: %{x}<extra></extra>",
        ))
        fig.update_layout(
            **CHART_LAYOUT,
            legend=LEGEND_STYLE,
            height=max(380, len(inv_s)*34),
            xaxis=dict(title="Units in stock", gridcolor="#f0f4f8"),
            yaxis=dict(tickfont_size=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        display = inv[["name","category","brand","unit","stock","reorder_pt","max_stock","sell_price","status"]].copy()
        display.columns = ["Product","Category","Brand","Unit","Stock","Reorder Pt","Max","Sell $","Status"]
        st.dataframe(display, use_container_width=True, hide_index=True)

    with tab2:
        pmap = get_product_map()
        if not pmap:
            st.warning("No products found.")
            return
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("#### Restock")
            rs_prod = st.selectbox("Product", list(pmap.keys()), key="rs_prod")
            rs_qty  = st.number_input("New Total Quantity", min_value=0, step=1, key="rs_qty")
            rs_note = st.text_input("Reason / Supplier", key="rs_note")
            if st.button("Update Stock", type="primary", key="rs_btn"):
                set_stock(pmap[rs_prod], rs_qty, "restock", rs_note or "Manual restock")
                st.success(f"Stock for '{rs_prod}' set to {rs_qty}.")
                st.rerun()
        with c2:
            st.markdown("#### Adjustment")
            aj_prod = st.selectbox("Product", list(pmap.keys()), key="aj_prod")
            aj_type = st.selectbox("Type", ["Add (received)","Remove (damaged/lost)","Write-off"], key="aj_type")
            aj_qty  = st.number_input("Quantity", min_value=1, step=1, key="aj_qty")
            aj_note = st.text_input("Reason", key="aj_note")
            if st.button("Apply Adjustment", type="primary", key="aj_btn"):
                delta = aj_qty if "Add" in aj_type else -aj_qty
                move  = "received" if "Add" in aj_type else ("damage" if "damaged" in aj_type else "writeoff")
                adjust_stock(pmap[aj_prod], delta, move, aj_note or aj_type)
                st.success(f"Adjustment of {delta:+d} applied to '{aj_prod}'.")
                st.rerun()

    with tab3:
        log = qdf("""
            SELECT sl.logged_at, p.name AS product, sl.type,
                   sl.qty_change, sl.qty_after, sl.reason
            FROM stock_log sl JOIN products p ON sl.product_id=p.id
            ORDER BY sl.logged_at DESC LIMIT 200
        """)
        if len(log):
            st.dataframe(log.rename(columns={
                "logged_at":"Timestamp","product":"Product","type":"Type",
                "qty_change":"Change","qty_after":"Stock After","reason":"Reason"}),
                use_container_width=True, hide_index=True)
        else:
            st.info("No movement history yet.")


# ── Sales Transactions ───────────────────────────────────
def page_sales():
    page_header("Sales Transactions")
    tab1, tab2 = st.tabs(["  Record Sale  ","  Sales History  "])

    with tab1:
        pmap = get_product_map()
        if not pmap:
            st.warning("Add products first.")
            return
        c1,c2 = st.columns([2,1])
        with c1:
            s_prod  = st.selectbox("Product *", list(pmap.keys()))
            pid     = pmap[s_prod]
            inv_row = qdf("SELECT sell_price, COALESCE(i.quantity,0) AS stock FROM products p LEFT JOIN inventory i ON p.id=i.product_id WHERE p.id=?", (pid,))
            curr_stock = int(inv_row["stock"].iloc[0]) if len(inv_row) else 0
            def_price  = float(inv_row["sell_price"].iloc[0]) if len(inv_row) else 0.0
            s_qty   = st.number_input(f"Quantity (available: {curr_stock})", min_value=1, max_value=max(1,curr_stock), step=1)
            s_price = st.number_input("Unit Price ($)", min_value=0.01, value=def_price, step=0.01)
            s_date  = st.date_input("Sale Date", value=date.today())
        with c2:
            s_customer = st.text_input("Customer Name")
            s_notes    = st.text_area("Notes", height=90)
            st.metric("Total Amount", f"${s_qty * s_price:,.2f}")

        if curr_stock == 0:
            st.error("This product is currently out of stock.")

        if st.button("Record Sale", type="primary"):
            if s_qty > curr_stock:
                st.error("Insufficient stock.")
            else:
                record_sale(pid, s_qty, s_price, s_date, s_customer, s_notes)
                st.success(f"Sale recorded: {s_qty} x {s_prod} = ${s_qty*s_price:,.2f}")
                st.rerun()

    with tab2:
        days_back = st.selectbox("Period", [7,30,60,90,0],
                                 format_func=lambda x: f"Last {x} days" if x else "All time")
        sales = get_sales(days_back if days_back else None)
        if len(sales):
            c1,c2,c3 = st.columns(3)
            c1.metric("Transactions", len(sales["transaction_id"].unique()))
            c2.metric("Units Sold",   int(sales["quantity"].sum()))
            c3.metric("Revenue",      f"${sales['total'].sum():,.2f}")
            st.dataframe(
                sales[["sale_date","product","category","quantity","unit_price","total","customer"]]
                     .rename(columns={"sale_date":"Date","product":"Product","category":"Cat",
                                      "quantity":"Qty","unit_price":"Unit $","total":"Total $",
                                      "customer":"Customer"}),
                use_container_width=True, hide_index=True)
        else:
            st.info("No sales in this period.")


# ── Demand Forecasting ────────────────────────────────────
def page_forecast():
    page_header("Demand Forecasting")

    pmap = get_product_map()
    if not pmap:
        st.warning("No products yet.")
        return

    cS, cD = st.columns([3,1])
    sel_prod   = cS.selectbox("Select product", list(pmap.keys()))
    days_ahead = cD.selectbox("Horizon", [14,30,60,90], index=1,
                              format_func=lambda x: f"{x} days")
    pid = pmap[sel_prod]
    hist, forecast, metrics, avg_daily = run_forecast(pid, days_ahead)

    if hist is None:
        st.warning("Not enough sales data (minimum 5 sale days required).")
        return

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Model Accuracy ",         f"{metrics['r2']:.3f}")
    m2.metric("Avg Prediction Error",  f"{metrics['mae']:.2f} units")
    m3.metric("Avg Daily Demand", f"{metrics['avg_daily']:.1f} units")
    m4.metric("Trend",            metrics["trend"])
    m5.metric("Data Points",      f"{metrics['n_days']} days")

    st.markdown("---")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hist["date"], y=hist["quantity"], name="Actual sales",
        marker_color="rgba(59,100,220,0.30)",
        hovertemplate="%{x|%b %d}: %{y} units<extra>Actual</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["predicted"],
        mode="lines", name="Regression line",
        line=dict(color=COLOR_PRIMARY, width=2.5),
        hovertemplate="%{x|%b %d}: %{y:.1f}<extra>Regression</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=forecast["date"], y=forecast["forecasted_qty"],
        mode="lines+markers", name=f"{days_ahead}-day forecast",
        line=dict(color=COLOR_SUCCESS, width=2.5, dash="dash"),
        marker=dict(size=5, color=COLOR_SUCCESS),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.07)",
        hovertemplate="%{x|%b %d}: %{y:.1f}<extra>Forecast</extra>",
    ))

    # Use add_shape instead of add_vline — avoids Plotly's internal
    # Timestamp arithmetic which raises TypeError on string x-values
    split_x = hist["date"].max().isoformat()
    fig.add_shape(
        type="line",
        x0=split_x, x1=split_x,
        y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#94a3b8", dash="dot", width=1.5),
    )
    fig.add_annotation(
        x=split_x, y=1.02,
        xref="x", yref="paper",
        text="Forecast start",
        showarrow=False,
        font=dict(size=11, color="#94a3b8"),
        xanchor="left",
    )

    fig.update_layout(**CHART_LAYOUT, legend=LEGEND_STYLE, height=380,
                      yaxis=dict(title="Units sold / forecasted", gridcolor="#f0f4f8"),
                      xaxis=dict(title="Date", gridcolor="#f0f4f8"))
    st.plotly_chart(fig, use_container_width=True)

    cA, cB = st.columns(2)
    with cA:
        section_title("Forecast summary by week")
        forecast["week"] = forecast["date"].dt.to_period("W").astype(str)
        weekly = (forecast.groupby("week")["forecasted_qty"]
                          .agg(total="sum", daily_avg="mean")
                          .reset_index()
                          .head(int(np.ceil(days_ahead/7))))
        weekly.columns = ["Week","Forecasted Units","Daily Avg"]
        weekly["Forecasted Units"] = weekly["Forecasted Units"].round(1)
        weekly["Daily Avg"]        = weekly["Daily Avg"].round(2)
        st.dataframe(weekly, use_container_width=True, hide_index=True)

    with cB:
        section_title("Inventory vs. forecast")
        inv_row    = qdf("SELECT COALESCE(i.quantity,0) AS stock, p.reorder_pt FROM products p LEFT JOIN inventory i ON p.id=i.product_id WHERE p.id=?", (pid,))
        curr_stock = int(inv_row["stock"].iloc[0])    if len(inv_row) else 0
        reorder_pt = int(inv_row["reorder_pt"].iloc[0]) if len(inv_row) else 0
        total_fc   = float(forecast["forecasted_qty"].sum())
        days_cover = round(curr_stock / avg_daily) if avg_daily > 0 else 999
        suggest    = max(0, int(total_fc) - curr_stock + reorder_pt)

        st.metric("Current Stock",         f"{curr_stock} units")
        st.metric("Total Forecast Demand", f"{total_fc:.0f} units")
        st.metric("Coverage",              f"{days_cover} days",
                  delta="Adequate" if days_cover >= days_ahead else "May run short",
                  delta_color="normal" if days_cover >= days_ahead else "inverse")
        st.metric("Suggested Order Qty",   f"{suggest} units" if suggest > 0 else "No order needed")


# ── Stockout Alerts ───────────────────────────────────────
def page_stockout():
    page_header("Stockout Alerts & Predictions")

    inv = get_inventory()
    critical, warning, watch, ok = [], [], [], []

    for _, row in inv.iterrows():
        pid   = int(row["id"])
        stock = int(row["stock"])
        days  = predict_stockout_days(pid, stock)
        entry = {"name": row["name"], "category": row["category"],
                 "stock": stock, "unit": row["unit"],
                 "rop": int(row["reorder_pt"]), "days": days, "status": row["status"]}
        if row["status"]   == "Out of Stock": critical.append(entry)
        elif row["status"] == "Low Stock":    warning.append(entry)
        elif days is not None and days <= 14: watch.append(entry)
        else:                                 ok.append(entry)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Out of Stock",     len(critical), delta_color="inverse")
    c2.metric("Low Stock",        len(warning),  delta_color="inverse")
    c3.metric("Stockout < 14d",   len(watch),    delta_color="inverse")
    c4.metric("Sufficient Stock", len(ok))
    st.markdown("---")

    def render(e, cls):
        days_str = (f"Estimated stockout in <b>{e['days']} days</b>"
                    if e["days"] else "Insufficient demand data")
        st.markdown(f"""
        <div class="{cls}">
          <div class="alert-title">{e['name']} &nbsp;·&nbsp; <span style="font-weight:400">{e['category']}</span></div>
          <div class="alert-body">
            Stock: <b>{e['stock']} {e['unit']}</b>
            &nbsp;·&nbsp; Reorder point: {e['rop']}
            &nbsp;·&nbsp; {days_str}
          </div>
        </div>""", unsafe_allow_html=True)

    if critical:
        section_title("Out of Stock — Immediate Action Required")
        for e in critical: render(e, "alert-critical")
    if warning:
        section_title("Low Stock — Order Soon")
        for e in warning: render(e, "alert-warning")
    if watch:
        section_title("Approaching Stockout — Monitor Closely")
        for e in watch: render(e, "alert-warning")
    if not (critical or warning or watch):
        st.markdown('<div class="alert-ok"><div class="alert-title">No urgent stockout risks detected</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    section_title("Stock coverage — all products")
    all_e = critical + warning + watch + ok
    cdf   = pd.DataFrame(all_e)
    cdf["days_display"] = cdf["days"].fillna(90).clip(upper=90)
    cdf["color"] = cdf.apply(lambda r:
        COLOR_DANGER  if r["status"]=="Out of Stock" else
        COLOR_WARNING if r["status"]=="Low Stock"    else
        "#f97316"     if r["days_display"]<=14       else
        COLOR_SUCCESS, axis=1)
    cdf = cdf.sort_values("days_display", ascending=True)
    fig = go.Figure(go.Bar(
        y=cdf["name"], x=cdf["days_display"], orientation="h",
        marker_color=cdf["color"],
        text=cdf["days"].apply(lambda d: f"{int(d)}d" if pd.notnull(d) else "—"),
        textposition="outside",
        hovertemplate="%{y}: %{x:.0f} days coverage<extra></extra>",
    ))
    fig.add_vline(x=14, line=dict(color=COLOR_WARNING, dash="dash", width=1.5),
                  annotation_text="14-day threshold", annotation_font_size=10)
    fig.update_layout(**CHART_LAYOUT, legend=LEGEND_STYLE,
                      height=max(380, len(cdf)*32),
                      xaxis=dict(range=[0,95], gridcolor="#f0f4f8",
                                 title="Estimated days until stockout"),
                      yaxis=dict(tickfont_size=10))
    st.plotly_chart(fig, use_container_width=True)


# ── Sales Analysis ─────────────────────────────────────────
def page_analysis():
    page_header("Sales Analysis")

    period = st.selectbox("Analysis period", [30,60,90],
                          format_func=lambda x: f"Last {x} days", index=1)
    sales  = get_sales(period)
    if not len(sales):
        st.info("No sales data for this period.")
        return

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Transactions",    len(sales["transaction_id"].unique()))
    c2.metric("Units Sold",      int(sales["quantity"].sum()))
    c3.metric("Total Revenue",   f"${sales['total'].sum():,.2f}")
    c4.metric("Avg Order Value", f"${sales.groupby('transaction_id')['total'].sum().mean():,.2f}")

    st.markdown("---")
    tab1,tab2,tab3,tab4 = st.tabs([
        "  Top Products  ","  Category Analysis  ","  Sales Trend  ","  Revenue Heatmap  "
    ])

    with tab1:
        top = (sales.groupby("product")
                    .agg(units=("quantity","sum"), revenue=("total","sum"))
                    .sort_values("revenue", ascending=False).head(10).reset_index())
        cA,cB = st.columns(2)
        with cA:
            fig = go.Figure(go.Bar(
                y=top["product"], x=top["revenue"], orientation="h",
                marker_color=COLOR_PRIMARY,
                text=top["revenue"].apply(lambda v: f"${v:,.0f}"), textposition="outside",
                hovertemplate="%{y}: $%{x:,.0f}<extra></extra>",
            ))
            fig.update_layout(**CHART_LAYOUT, legend=LEGEND_STYLE, height=340,
                              xaxis=dict(gridcolor="#f0f4f8", title="Revenue ($)"),
                              title=dict(text="By revenue", font_size=13))
            st.plotly_chart(fig, use_container_width=True)
        with cB:
            fig2 = go.Figure(go.Bar(
                y=top["product"], x=top["units"], orientation="h",
                marker_color=COLOR_SUCCESS,
                text=top["units"].astype(str), textposition="outside",
                hovertemplate="%{y}: %{x} units<extra></extra>",
            ))
            fig2.update_layout(**CHART_LAYOUT, legend=LEGEND_STYLE, height=340,
                               xaxis=dict(gridcolor="#f0f4f8", title="Units sold"),
                               title=dict(text="By units sold", font_size=13))
            st.plotly_chart(fig2, use_container_width=True)
        top["revenue"] = top["revenue"].apply(lambda v: f"${v:,.2f}")
        st.dataframe(top.rename(columns={"product":"Product","units":"Units","revenue":"Revenue"}),
                     use_container_width=True, hide_index=True)

    with tab2:
        cat = (sales.groupby("category")
                    .agg(units=("quantity","sum"), revenue=("total","sum"),
                         transactions=("transaction_id","nunique"))
                    .sort_values("revenue", ascending=False).reset_index())
        cA,cB = st.columns([3,2])
        with cA:
            fig = go.Figure(go.Bar(
                x=cat["category"], y=cat["revenue"],
                marker_color=[CATEGORY_COLORS.get(c,"#94a3b8") for c in cat["category"]],
                text=cat["revenue"].apply(lambda v: f"${v:,.0f}"), textposition="outside",
                hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
            ))
            fig.update_layout(**CHART_LAYOUT, legend=LEGEND_STYLE, height=300,
                              yaxis=dict(gridcolor="#f0f4f8", title="Revenue ($)"),
                              xaxis=dict(tickfont_size=11),
                              title=dict(text="Revenue by category", font_size=13))
            st.plotly_chart(fig, use_container_width=True)
        with cB:
            fig2 = go.Figure(go.Pie(
                labels=cat["category"], values=cat["revenue"], hole=0.55,
                marker_colors=[CATEGORY_COLORS.get(c,"#94a3b8") for c in cat["category"]],
                hovertemplate="%{label}: $%{value:,.0f} (%{percent})<extra></extra>",
            ))
            fig2.update_layout(**CHART_LAYOUT, legend=dict(font_size=10), height=300,
                               title=dict(text="Share of revenue", font_size=13))
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        daily = sales.groupby("sale_date").agg(revenue=("total","sum"), units=("quantity","sum")).reset_index()
        daily["sale_date"] = pd.to_datetime(daily["sale_date"])
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(
            x=daily["sale_date"], y=daily["revenue"], name="Revenue",
            marker_color="rgba(59,100,220,0.55)",
            hovertemplate="%{x|%b %d}: $%{y:,.0f}<extra></extra>",
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=daily["sale_date"], y=daily["units"], mode="lines", name="Units",
            line=dict(color=COLOR_SUCCESS, width=2),
            hovertemplate="%{x|%b %d}: %{y}<extra></extra>",
        ), secondary_y=True)
        fig.update_layout(**CHART_LAYOUT, legend=LEGEND_STYLE, height=320,
                          title=dict(text="Daily revenue and units", font_size=13))
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=False, gridcolor="#f0f4f8")
        fig.update_yaxes(title_text="Units",       secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        sales["week"] = pd.to_datetime(sales["sale_date"]).dt.strftime("W%U")
        heat = sales.groupby(["product","week"])["total"].sum().unstack(fill_value=0)
        if len(heat) > 1:
            fig = go.Figure(go.Heatmap(
                z=heat.values, x=heat.columns.tolist(), y=heat.index.tolist(),
                colorscale="Blues",
                hovertemplate="Product: %{y}<br>Week: %{x}<br>Revenue: $%{z:,.0f}<extra></extra>",
            ))
            fig.update_layout(**CHART_LAYOUT, legend=LEGEND_STYLE,
                              height=max(300, len(heat)*24),
                              xaxis_title="Week", yaxis=dict(tickfont_size=9),
                              title=dict(text="Revenue heatmap — product vs. week", font_size=13))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need more product variety for heatmap.")


# ── Market Basket Analysis ─────────────────────────────────
def page_basket():
    page_header("Market Basket Analysis")

    st.markdown("---")
    cA, cB, cC = st.columns(3)
    min_support    = cA.slider("Min. Support",    0.005, 0.10, 0.01, 0.005,
                               help="Fraction of transactions containing the itemset")
    min_confidence = cB.slider("Min. Confidence", 0.10,  0.90, 0.20, 0.05,
                               help="P(B | A) — how often the rule is correct")
    days_back      = cC.selectbox("Transaction window", [30,60,90], index=1,
                                  format_func=lambda x: f"Last {x} days")

    with st.spinner("Mining association rules..."):
        freq_sets, rules, n_txns = get_association_rules(min_support=min_support, days_back=days_back)

    if rules is None:
        st.warning("Not enough multi-item transaction data. Lower the minimum support or extend the window.")
        return

    rules_f = rules[rules["confidence"] >= min_confidence].copy()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Transactions analysed", f"{n_txns:,}")
    c2.metric("Frequent itemsets",     len(freq_sets))
    c3.metric("Rules found",           len(rules))
    c4.metric("Rules after filter",    len(rules_f))
    st.markdown("---")

    if not len(rules_f):
        st.info("No rules meet the current confidence threshold. Try lowering it.")
        return

    tab1, tab2, tab3 = st.tabs([
        "  Top Associations  ",
        "  Lift vs. Confidence  ",
        "  Frequent Itemsets  ",
    ])

    with tab1:
        section_title("Customers who buy X are likely to also buy Y — sorted by lift")
        for _, row in rules_f.head(20).iterrows():
            lift_col = (COLOR_SUCCESS if row["lift"] >= 2.0 else
                        COLOR_PRIMARY if row["lift"] >= 1.5 else "#94a3b8")
            st.markdown(f"""
            <div class="rule-card">
              <div class="rule-lhs">{row['antecedents_str']}</div>
              <div class="rule-arrow">&#8594;</div>
              <div class="rule-rhs">{row['consequents_str']}</div>
              <div class="rule-stats">
                <div class="rule-stat">
                  <div class="rule-stat-val" style="color:{lift_col};">{row['lift']:.2f}</div>
                  <div class="rule-stat-lbl">Lift</div>
                </div>
                <div class="rule-stat">
                  <div class="rule-stat-val">{row['confidence']:.0%}</div>
                  <div class="rule-stat-lbl">Confidence</div>
                </div>
                <div class="rule-stat">
                  <div class="rule-stat-val">{row['support']:.3f}</div>
                  <div class="rule-stat-lbl">Support</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        section_title("Full rules table")
        disp = rules_f[["antecedents_str","consequents_str","support","confidence","lift"]].copy()
        disp.columns = ["If customer buys","They also buy","Support","Confidence","Lift"]
        disp["Support"]    = disp["Support"].apply(lambda v: f"{v:.3f}")
        disp["Confidence"] = disp["Confidence"].apply(lambda v: f"{v:.1%}")
        disp["Lift"]       = disp["Lift"].apply(lambda v: f"{v:.2f}")
        st.dataframe(disp, use_container_width=True, hide_index=True)

    with tab2:
        section_title("Lift vs. confidence — bubble size = support")
        fig = go.Figure(go.Scatter(
            x=rules_f["confidence"],
            y=rules_f["lift"],
            mode="markers",
            marker=dict(
                size=(rules_f["support"] * 800).clip(8, 40),
                color=rules_f["lift"],
                colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Lift", thickness=12),
                line=dict(width=1, color="rgba(59,100,220,0.4)"),
            ),
            text=rules_f["antecedents_str"] + " -> " + rules_f["consequents_str"],
            hovertemplate="%{text}<br>Confidence: %{x:.1%}<br>Lift: %{y:.2f}<extra></extra>",
        ))
        fig.add_hline(y=1.0, line=dict(color="#94a3b8", dash="dot", width=1.5),
                      annotation_text="Lift = 1 (random chance)", annotation_font_size=10)
        fig.update_layout(**CHART_LAYOUT, legend=LEGEND_STYLE, height=420,
                          xaxis=dict(title="Confidence", tickformat=".0%", gridcolor="#f0f4f8"),
                          yaxis=dict(title="Lift", gridcolor="#f0f4f8"),
                          title=dict(text="Rule quality — lift vs. confidence (bubble = support)", font_size=13))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="alert-info">
          <div class="alert-body">
            <b>Lift > 1</b> — products appear together more often than chance.<br>
            <b>High confidence</b> — buying the antecedent reliably leads to buying the consequent.<br>
            <b>Bubble size</b> — frequency across all transactions (support).
          </div>
        </div>""", unsafe_allow_html=True)

    with tab3:
        section_title("Most frequent itemsets")
        top_sets = freq_sets.sort_values("support", ascending=False).head(30).copy()
        top_sets["itemsets_str"] = top_sets["itemsets"].apply(lambda x: ", ".join(sorted(x)))
        top_sets["n_items"]      = top_sets["itemsets"].apply(len)
        top_sets["support_pct"]  = (top_sets["support"] * 100).round(2).astype(str) + "%"
        fig = go.Figure(go.Bar(
            y=top_sets["itemsets_str"].str[:50],
            x=top_sets["support"],
            orientation="h",
            marker_color=[
                COLOR_PRIMARY if n == 1 else
                COLOR_PURPLE  if n == 2 else
                COLOR_TEAL
                for n in top_sets["n_items"]
            ],
            text=top_sets["support_pct"], textposition="outside",
            hovertemplate="%{y}<br>Support: %{x:.3f}<extra></extra>",
        ))
        fig.update_layout(**CHART_LAYOUT, legend=LEGEND_STYLE,
                          height=max(320, len(top_sets)*28),
                          xaxis=dict(title="Support", gridcolor="#f0f4f8"),
                          yaxis=dict(tickfont_size=10),
                          title=dict(text="Top 30 frequent itemsets", font_size=13))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        <div class="alert-info">
          <div class="alert-body">
            Blue = single item &nbsp;·&nbsp; Purple = 2-item pairs &nbsp;·&nbsp; Teal = 3-item sets
          </div>
        </div>""", unsafe_allow_html=True)


# ── Reports ───────────────────────────────────────────────
def page_reports():
    page_header("Reports")
    tab1,tab2,tab3 = st.tabs(["  Inventory Report  ","  Sales Report  ","  Forecast Report  "])

    def to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    with tab1:
        inv = get_inventory()
        inv["stock_value"] = (inv["stock"] * inv["sell_price"]).round(2)
        report = inv[["name","category","brand","unit","stock","reorder_pt",
                       "max_stock","sell_price","cost_price","stock_value","status"]].copy()
        report.columns = ["Product","Category","Brand","Unit","Stock","Reorder Pt",
                           "Max Stock","Sell $","Cost $","Stock Value","Status"]
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Products",    len(report))
        c2.metric("Total Stock Value", f"${report['Stock Value'].sum():,.2f}")
        c3.metric("Low / Out of Stock",len(report[report["Status"].isin(["Low Stock","Out of Stock"])]))
        st.dataframe(report, use_container_width=True, hide_index=True)
        st.download_button("Download Inventory Report (CSV)", to_csv(report),
                           f"inventory_report_{date.today()}.csv", "text/csv", type="primary")

    with tab2:
        days = st.selectbox("Period", [7,30,60,90,0],
                            format_func=lambda x: f"Last {x} days" if x else "All time")
        sales = get_sales(days if days else None)
        if len(sales):
            c1,c2,c3 = st.columns(3)
            c1.metric("Transactions",  len(sales["transaction_id"].unique()))
            c2.metric("Units Sold",    int(sales["quantity"].sum()))
            c3.metric("Total Revenue", f"${sales['total'].sum():,.2f}")
            st.dataframe(sales.drop(columns=["id"], errors="ignore"),
                         use_container_width=True, hide_index=True)
            st.download_button("Download Sales Report (CSV)", to_csv(sales),
                               f"sales_report_{date.today()}.csv", "text/csv", type="primary")
        else:
            st.info("No sales for this period.")

    with tab3:
        pmap = get_product_map()
        sel  = st.selectbox("Product", list(pmap.keys()))
        _, forecast, metrics, _ = run_forecast(pmap[sel], 30)
        if forecast is not None:
            inv_row  = qdf("SELECT COALESCE(i.quantity,0) AS stock FROM products p LEFT JOIN inventory i ON p.id=i.product_id WHERE p.id=?", (pmap[sel],))
            curr_stk = int(inv_row["stock"].iloc[0]) if len(inv_row) else 0
            c1,c2,c3 = st.columns(3)
            c1.metric("R² Score", metrics["r2"])
            c2.metric("MAE",      metrics["mae"])
            c3.metric("Trend",    metrics["trend"])
            forecast["product"]       = sel
            forecast["current_stock"] = curr_stk
            forecast["model"]         = "Simple Linear Regression"
            forecast["generated"]     = datetime.now().strftime("%Y-%m-%d %H:%M")
            st.dataframe(forecast, use_container_width=True, hide_index=True)
            st.download_button("Download Forecast Report (CSV)", to_csv(forecast),
                               f"forecast_{sel.replace(' ','_')}_{date.today()}.csv",
                               "text/csv", type="primary")
        else:
            st.warning("Insufficient sales data for this product.")


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
PAGES = {
    "Dashboard":               page_dashboard,
    "Product Management":      page_products,
    "Inventory Monitoring":    page_inventory,
    "Sales Transactions":      page_sales,
    "Demand Forecasting":      page_forecast,
    "Stockout Alerts":         page_stockout,
    "Sales Analysis":          page_analysis,
    "Market Basket Analysis":  page_basket,
    "Reports":                 page_reports,
}


def main():
    inject_css()

    # ── Auth gate ──────────────────────────────────────
    if not st.session_state.get("logged_in"):
        render_login()
        st.stop()   # nothing below runs until logged in

    # ── Seeding (runs once after first login) ──────────
    seed_walmart_data()

    # ── Sidebar ────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:22px 18px 16px;border-bottom:1px solid rgba(99,132,199,0.18);margin-bottom:8px;">
          <div style="font-size:18px;font-weight:700;letter-spacing:-0.5px;
               background:linear-gradient(90deg,#93c5fd,#c4b5fd);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            IntelliStock
          </div>
          <div style="font-size:10px;color:#3a506b;margin-top:4px;
               text-transform:uppercase;letter-spacing:1.2px;">
            Smart Inventory v1.0
          </div>
        </div>
        <div style="padding:10px 18px 12px;border-bottom:1px solid rgba(99,132,199,0.12);
             margin-bottom:8px;">
          <div style="font-size:12px;color:#c8d8f0;font-weight:500;">
            {st.session_state['name']}
          </div>
          <div style="font-size:10px;color:#3a506b;margin-top:2px;text-transform:uppercase;
               letter-spacing:0.8px;">
            {st.session_state['role']}
          </div>
        </div>
        """, unsafe_allow_html=True)

        selected = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")

        st.markdown("---")

        inv_df = get_inventory()
        low_n  = len(inv_df[inv_df["status"].isin(["Low Stock","Out of Stock"])])
        st.markdown(f"""
        <div style="padding:10px 6px 4px;font-size:11px;">
          <div style="margin-bottom:7px;display:flex;justify-content:space-between;">
            <span style="color:#3a506b;">Products</span>
            <span style="color:#c8d8f0;font-weight:600;">{len(inv_df)}</span>
          </div>
          <div style="margin-bottom:7px;display:flex;justify-content:space-between;">
            <span style="color:#3a506b;">Alerts</span>
            <span style="color:{'#ef4444' if low_n else '#10b981'};font-weight:600;">{low_n}</span>
          </div>
          <div style="display:flex;justify-content:space-between;">
            <span style="color:#3a506b;">Updated</span>
            <span style="color:#c8d8f0;">{datetime.now().strftime('%H:%M')}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("Sign Out", use_container_width=True):
            for key in ["logged_in", "username", "name", "role"]:
                st.session_state.pop(key, None)
            st.rerun()


    PAGES[selected]()


if __name__ == "__main__":
    main()