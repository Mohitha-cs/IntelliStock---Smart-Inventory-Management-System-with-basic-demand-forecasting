"""
One-time IntelliStock sales refresh.

Run this file from the project directory containing:
    stockwise.db

Command:
    python3 refresh_sales.py
"""

import sqlite3
import random
import uuid
from datetime import date, timedelta
from pathlib import Path

# Always use the database located beside this script.
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "stockwise.db"

DAYS_TO_GENERATE = 90
RANDOM_SEED = 20260912

# This marker prevents accidental duplicate refreshes.
REFRESH_MARKER = "sales_refresh_90_days_2026_09_12"


def get_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at:\n{DB_PATH}\n\n"
            "Save this script in the same folder as stockwise.db."
        )

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_refresh_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS refresh_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            marker TEXT NOT NULL UNIQUE,
            executed_at TEXT NOT NULL DEFAULT (datetime('now')),
            rows_inserted INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    connection.commit()


def already_refreshed(connection):
    row = connection.execute(
        """
        SELECT 1
        FROM refresh_history
        WHERE marker = ?
        LIMIT 1
        """,
        (REFRESH_MARKER,),
    ).fetchone()

    return row is not None


def load_products(connection):
    rows = connection.execute(
        """
        SELECT
            id,
            name,
            category,
            sell_price,
            active
        FROM products
        WHERE active = 1
        ORDER BY id
        """
    ).fetchall()

    if not rows:
        raise RuntimeError(
            "No active products were found in the database. "
            "Run the original product seeding process first."
        )

    return rows


def category_weight(category):
    """
    Approximate Walmart-style demand weights.
    Higher values produce more frequent sales.
    """

    weights = {
        "Grocery": 1.45,
        "Beverages": 1.35,
        "Baby & Toddler": 1.15,
        "Personal Care": 1.10,
        "Home & Cleaning": 1.05,
        "Pharmacy": 0.95,
        "Clothing": 0.75,
        "Sports & Fitness": 0.65,
        "Electronics": 0.55,
        "Auto": 0.50,
    }

    return weights.get(category, 0.85)


def product_base_demand(category):
    """
    Average number of units sold when a product appears in a transaction.
    """

    demand = {
        "Grocery": 2.2,
        "Beverages": 2.4,
        "Baby & Toddler": 1.7,
        "Personal Care": 1.6,
        "Home & Cleaning": 1.8,
        "Pharmacy": 1.4,
        "Clothing": 1.2,
        "Sports & Fitness": 1.1,
        "Electronics": 1.0,
        "Auto": 1.0,
    }

    return demand.get(category, 1.3)


def generate_sales(products):
    random.seed(RANDOM_SEED)

    today = date.today()
    start_day = today - timedelta(days=DAYS_TO_GENERATE - 1)

    # Convert SQLite rows to ordinary dictionaries.
    product_data = []

    for product in products:
        product_data.append(
            {
                "id": int(product["id"]),
                "name": product["name"],
                "category": product["category"] or "General",
                "price": float(product["sell_price"] or 0),
                "weight": category_weight(product["category"]),
            }
        )

    total_weight = sum(item["weight"] for item in product_data)

    sales_rows = []

    for day_offset in range(DAYS_TO_GENERATE):
        sale_day = start_day + timedelta(days=day_offset)

        # Slightly different demand by weekday.
        weekday = sale_day.weekday()

        if weekday in (4, 5):       # Friday and Saturday
            transaction_count = random.randint(45, 75)
        elif weekday == 6:          # Sunday
            transaction_count = random.randint(35, 65)
        elif weekday == 0:          # Monday
            transaction_count = random.randint(30, 55)
        else:
            transaction_count = random.randint(35, 65)

        for _ in range(transaction_count):
            transaction_id = str(uuid.uuid4())

            # One to four distinct products per transaction.
            basket_size = random.choices(
                population=[1, 2, 3, 4],
                weights=[48, 32, 15, 5],
                k=1,
            )[0]

            basket = random.choices(
                population=product_data,
                weights=[
                    item["weight"] / total_weight
                    for item in product_data
                ],
                k=basket_size,
            )

            # Remove duplicate product IDs from the basket.
            unique_products = {}
            for item in basket:
                unique_products[item["id"]] = item

            for item in unique_products.values():
                base_demand = product_base_demand(item["category"])

                # Most sales are 1–3 units, with occasional larger purchases.
                quantity = max(
                    1,
                    int(
                        round(
                            random.gauss(
                                mu=base_demand,
                                sigma=max(0.5, base_demand * 0.45),
                            )
                        )
                    ),
                )

                # Small price variation represents discounts/promotions.
                price_multiplier = random.choices(
                    population=[1.00, 0.95, 0.90, 1.05],
                    weights=[70, 15, 10, 5],
                    k=1,
                )[0]

                unit_price = round(
                    item["price"] * price_multiplier,
                    2,
                )

                total = round(quantity * unit_price, 2)

                sales_rows.append(
                    (
                        transaction_id,
                        item["id"],
                        quantity,
                        unit_price,
                        total,
                        sale_day.isoformat(),
                        "",
                        "Generated Walmart-style historical sales",
                    )
                )

    return sales_rows


def insert_sales(connection, sales_rows):
    connection.executemany(
        """
        INSERT INTO sales (
            transaction_id,
            product_id,
            quantity,
            unit_price,
            total,
            sale_date,
            customer,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        sales_rows,
    )

    connection.execute(
        """
        INSERT INTO refresh_history (
            marker,
            rows_inserted
        )
        VALUES (?, ?)
        """,
        (REFRESH_MARKER, len(sales_rows)),
    )

    connection.commit()


def print_summary(connection, inserted_count):
    summary = connection.execute(
        """
        SELECT
            COUNT(*) AS total_sales_rows,
            MIN(sale_date) AS earliest_sale,
            MAX(sale_date) AS latest_sale,
            ROUND(SUM(total), 2) AS total_revenue
        FROM sales
        """
    ).fetchone()

    recent_summary = connection.execute(
        """
        SELECT
            COUNT(*) AS recent_rows,
            ROUND(SUM(total), 2) AS recent_revenue
        FROM sales
        WHERE sale_date >= date('now', '-90 days')
        """
    ).fetchone()

    print("\nSales refresh completed successfully.")
    print("----------------------------------")
    print(f"Database:              {DB_PATH}")
    print(f"New sales rows added:  {inserted_count}")
    print(f"Total sales rows:      {summary['total_sales_rows']}")
    print(f"Earliest sale date:    {summary['earliest_sale']}")
    print(f"Latest sale date:      {summary['latest_sale']}")
    print(f"Total revenue:         {summary['total_revenue']}")
    print()
    print("Last 90 days:")
    print(f"Sales rows:            {recent_summary['recent_rows']}")
    print(f"Revenue:               {recent_summary['recent_revenue']}")
    print("----------------------------------")


def main():
    print(f"Using database:\n{DB_PATH}\n")

    connection = get_connection()

    try:
        ensure_refresh_table(connection)

        if already_refreshed(connection):
            print(
                "This refresh has already been completed.\n"
                "No new rows were inserted.\n"
                f"Marker: {REFRESH_MARKER}"
            )
            return

        products = load_products(connection)

        print(f"Active products found: {len(products)}")
        print(f"Generating sales for the last {DAYS_TO_GENERATE} days...")

        sales_rows = generate_sales(products)

        if not sales_rows:
            raise RuntimeError("No sales rows were generated.")

        print(f"Generated {len(sales_rows)} sales rows.")
        print("Inserting sales into SQLite database...")

        insert_sales(connection, sales_rows)
        print_summary(connection, len(sales_rows))

    finally:
        connection.close()


if __name__ == "__main__":
    main()