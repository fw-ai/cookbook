#!/usr/bin/env python3
"""
Setup script to create sales.db and finance.db with planted data quality
issues and cross-system discrepancies for the reconciliation task.
All data is deterministic.
"""
import sqlite3
import os

DATA_DIR = '/app/data'

def create_sales_db():
    conn = sqlite3.connect(os.path.join(DATA_DIR, 'sales.db'))
    c = conn.cursor()

    c.execute('''CREATE TABLE customers (
        customer_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        region TEXT NOT NULL,
        signup_date TEXT NOT NULL
    )''')
    customers = [
        ('C001', 'Alice Johnson', 'alice.j@example.com', 'Northeast', '2022-06-15'),
        ('C002', 'Bob Martinez', 'bob.m@example.com', 'Southwest', '2022-07-20'),
        ('C003', 'Carol Williams', 'carol.w@example.com', 'Midwest', '2022-08-10'),
        ('C004', 'David Lee', 'david.l@example.com', 'West', '2022-09-01'),
        ('C005', 'Emma Davis', 'emma.d@example.com', 'Southeast', '2022-10-05'),
        ('C006', 'Frank Wilson', 'frank.w@example.com', 'Northeast', '2022-11-12'),
        ('C007', 'Grace Chen', 'grace.c@example.com', 'West', '2023-01-08'),
        ('C008', 'Henry Brown', 'henry.b@example.com', 'Midwest', '2023-02-14'),
        # Duplicate of C001 - typo in name, different email, same region
        ('C009', 'Alce Johnson', 'alice.johnson@example.com', 'Northeast', '2022-06-20'),
        # Duplicate of C004 - exact same name, different email, same region
        ('C010', 'David Lee', 'david.lee@example.com', 'West', '2022-09-05'),
        ('C011', 'Isabella Moore', 'isabella.m@example.com', 'Southeast', '2023-03-01'),
        ('C012', 'James Taylor', 'james.t@example.com', 'Southwest', '2023-03-15'),
    ]
    c.executemany('INSERT INTO customers VALUES (?,?,?,?,?)', customers)

    c.execute('''CREATE TABLE products (
        product_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        unit_price REAL NOT NULL
    )''')
    products = [
        ('P001', 'Widget Alpha', 'Hardware', 25.00),
        ('P002', 'Widget Beta', 'Hardware', 50.00),
        ('P003', 'Gadget Pro', 'Electronics', 150.00),
        ('P004', 'Gadget Lite', 'Electronics', 90.00),
        ('P005', 'Service Plan', 'Services', 60.00),
        ('P006', 'Premium Support', 'Services', 130.00),
    ]
    c.executemany('INSERT INTO products VALUES (?,?,?,?)', products)

    c.execute('''CREATE TABLE orders (
        order_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL,
        order_date TEXT NOT NULL,
        total_amount REAL NOT NULL,
        status TEXT NOT NULL DEFAULT 'completed',
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    )''')
    # Date format issues: O003, O008, O011, O016, O019 use MM/DD/YYYY
    # Amount errors: O007 (stated 236, items 230), O013 (stated 280, items 255)
    # Intentional surcharge: O018 (stated 605, items 580 - $25 express shipping)
    orders = [
        ('O001', 'C001', '2023-02-01', 100.00, 'completed'),
        ('O002', 'C002', '2023-02-03', 200.00, 'completed'),
        ('O003', 'C003', '02/05/2023', 150.00, 'completed'),       # date format issue
        ('O004', 'C004', '2023-02-08', 240.00, 'completed'),
        ('O005', 'C005', '2023-02-10', 160.00, 'completed'),
        ('O006', 'C006', '2023-02-12', 180.00, 'completed'),
        ('O007', 'C007', '2023-02-15', 236.00, 'completed'),       # ERROR: items sum to 230.00
        ('O008', 'C008', '02/18/2023', 150.00, 'completed'),       # date format issue
        ('O009', 'C001', '2023-02-20', 325.00, 'completed'),
        ('O010', 'C009', '2023-02-22', 165.00, 'completed'),       # order by duplicate customer
        ('O011', 'C004', '02/25/2023', 230.00, 'completed'),       # date format issue
        ('O012', 'C010', '2023-02-28', 210.00, 'completed'),       # order by duplicate customer
        ('O013', 'C002', '2023-03-02', 280.00, 'completed'),       # ERROR: items sum to 255.00
        ('O014', 'C011', '2023-03-05', 155.00, 'completed'),
        ('O015', 'C003', '2023-03-08', 350.00, 'completed'),
        ('O016', 'C005', '03/10/2023', 210.00, 'completed'),       # date format issue
        ('O017', 'C012', '2023-03-12', 220.00, 'completed'),
        ('O018', 'C006', '2023-03-15', 605.00, 'completed'),       # INTENTIONAL: $25 express shipping
        ('O019', 'C007', '03/18/2023', 150.00, 'completed'),       # date format issue
        ('O020', 'C008', '2023-03-20', 300.00, 'completed'),
    ]
    c.executemany('INSERT INTO orders VALUES (?,?,?,?,?)', orders)

    c.execute('''CREATE TABLE order_items (
        item_id TEXT PRIMARY KEY,
        order_id TEXT NOT NULL,
        product_id TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    )''')
    order_items = [
        # O001: 2*P001(25)+1*P002(50) = 100
        ('OI001', 'O001', 'P001', 2),
        ('OI002', 'O001', 'P002', 1),
        # O002: 1*P003(150)+1*P002(50) = 200
        ('OI003', 'O002', 'P003', 1),
        ('OI004', 'O002', 'P002', 1),
        # O003: 1*P004(90)+1*P005(60) = 150
        ('OI005', 'O003', 'P004', 1),
        ('OI006', 'O003', 'P005', 1),
        # O004: 1*P003(150)+1*P004(90) = 240
        ('OI007', 'O004', 'P003', 1),
        ('OI008', 'O004', 'P004', 1),
        # O005: 2*P002(50)+1*P005(60) = 160
        ('OI009', 'O005', 'P002', 2),
        ('OI010', 'O005', 'P005', 1),
        # O006: 1*P006(130)+2*P001(25) = 180
        ('OI011', 'O006', 'P006', 1),
        ('OI012', 'O006', 'P001', 2),
        # O007: 4*P001(25)+1*P006(130) = 230  (order says 236 -> ERROR)
        ('OI013', 'O007', 'P001', 4),
        ('OI014', 'O007', 'P006', 1),
        # O008: 1*P004(90)+1*P005(60) = 150
        ('OI015', 'O008', 'P004', 1),
        ('OI016', 'O008', 'P005', 1),
        # O009: 2*P003(150)+1*P001(25) = 325
        ('OI017', 'O009', 'P003', 2),
        ('OI018', 'O009', 'P001', 1),
        # O010: 1*P002(50)+1*P004(90)+1*P001(25) = 165
        ('OI019', 'O010', 'P002', 1),
        ('OI020', 'O010', 'P004', 1),
        ('OI021', 'O010', 'P001', 1),
        # O011: 3*P005(60)+1*P002(50) = 230
        ('OI022', 'O011', 'P005', 3),
        ('OI023', 'O011', 'P002', 1),
        # O012: 1*P003(150)+1*P005(60) = 210
        ('OI024', 'O012', 'P003', 1),
        ('OI025', 'O012', 'P005', 1),
        # O013: 2*P004(90)+1*P002(50)+1*P001(25) = 255  (order says 280 -> ERROR)
        ('OI026', 'O013', 'P004', 2),
        ('OI027', 'O013', 'P002', 1),
        ('OI028', 'O013', 'P001', 1),
        # O014: 1*P006(130)+1*P001(25) = 155
        ('OI029', 'O014', 'P006', 1),
        ('OI030', 'O014', 'P001', 1),
        # O015: 2*P003(150)+2*P001(25) = 350
        ('OI031', 'O015', 'P003', 2),
        ('OI032', 'O015', 'P001', 2),
        # O016: 1*P004(90)+2*P005(60) = 210
        ('OI033', 'O016', 'P004', 1),
        ('OI034', 'O016', 'P005', 2),
        # O017: 1*P006(130)+1*P004(90) = 220
        ('OI035', 'O017', 'P006', 1),
        ('OI036', 'O017', 'P004', 1),
        # O018: 3*P003(150)+1*P006(130) = 580  (order says 605 -> INTENTIONAL surcharge)
        ('OI037', 'O018', 'P003', 3),
        ('OI038', 'O018', 'P006', 1),
        # O019: 2*P002(50)+2*P001(25) = 150
        ('OI039', 'O019', 'P002', 2),
        ('OI040', 'O019', 'P001', 2),
        # O020: 1*P003(150)+1*P004(90)+1*P005(60) = 300
        ('OI041', 'O020', 'P003', 1),
        ('OI042', 'O020', 'P004', 1),
        ('OI043', 'O020', 'P005', 1),
    ]
    c.executemany('INSERT INTO order_items VALUES (?,?,?,?)', order_items)

    conn.commit()
    conn.close()


def create_finance_db():
    conn = sqlite3.connect(os.path.join(DATA_DIR, 'finance.db'))
    c = conn.cursor()

    c.execute('''CREATE TABLE accounts (
        account_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL
    )''')
    accounts = [
        ('ACC001', 'Sales Revenue', 'revenue'),
        ('ACC002', 'Accounts Receivable', 'asset'),
        ('ACC003', 'Cash', 'asset'),
        ('ACC004', 'Returns & Adjustments', 'contra_revenue'),
        ('ACC005', 'Operating Expenses', 'expense'),
    ]
    c.executemany('INSERT INTO accounts VALUES (?,?,?)', accounts)

    c.execute('''CREATE TABLE invoices (
        invoice_id TEXT PRIMARY KEY,
        order_id TEXT NOT NULL,
        customer_id TEXT NOT NULL,
        amount REAL NOT NULL,
        invoice_date TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'paid'
    )''')
    # Invoice amounts match order totals (including wrong ones - cascaded errors)
    # INV-021 and INV-022 reference non-existent orders (orphaned)
    invoices = [
        ('INV-001', 'O001', 'C001', 100.00, '2023-02-02', 'paid'),
        ('INV-002', 'O002', 'C002', 200.00, '2023-02-04', 'paid'),
        ('INV-003', 'O003', 'C003', 150.00, '2023-02-06', 'paid'),
        ('INV-004', 'O004', 'C004', 240.00, '2023-02-09', 'paid'),
        ('INV-005', 'O005', 'C005', 160.00, '2023-02-11', 'paid'),
        ('INV-006', 'O006', 'C006', 180.00, '2023-02-13', 'paid'),
        ('INV-007', 'O007', 'C007', 236.00, '2023-02-16', 'paid'),  # cascaded from O007 error
        ('INV-008', 'O008', 'C008', 150.00, '2023-02-19', 'paid'),
        ('INV-009', 'O009', 'C001', 325.00, '2023-02-21', 'paid'),
        ('INV-010', 'O010', 'C009', 165.00, '2023-02-23', 'paid'),
        ('INV-011', 'O011', 'C004', 230.00, '2023-02-26', 'paid'),
        ('INV-012', 'O012', 'C010', 210.00, '2023-03-01', 'paid'),
        ('INV-013', 'O013', 'C002', 280.00, '2023-03-03', 'paid'),  # cascaded from O013 error
        ('INV-014', 'O014', 'C011', 155.00, '2023-03-06', 'paid'),
        ('INV-015', 'O015', 'C003', 350.00, '2023-03-09', 'paid'),
        ('INV-016', 'O016', 'C005', 210.00, '2023-03-11', 'paid'),
        ('INV-017', 'O017', 'C012', 220.00, '2023-03-13', 'paid'),
        ('INV-018', 'O018', 'C006', 605.00, '2023-03-16', 'paid'),
        ('INV-019', 'O019', 'C007', 150.00, '2023-03-19', 'paid'),
        ('INV-020', 'O020', 'C008', 300.00, '2023-03-21', 'paid'),
        # Orphaned invoices - reference non-existent orders
        ('INV-021', 'O099', 'C002', 200.00, '2023-03-25', 'pending'),
        ('INV-022', 'O098', 'C005', 350.00, '2023-03-28', 'pending'),
    ]
    c.executemany('INSERT INTO invoices VALUES (?,?,?,?,?,?)', invoices)

    c.execute('''CREATE TABLE transactions (
        txn_id TEXT PRIMARY KEY,
        invoice_id TEXT,
        account_id TEXT NOT NULL,
        amount REAL NOT NULL,
        txn_date TEXT NOT NULL,
        type TEXT NOT NULL,
        reference TEXT,
        FOREIGN KEY (account_id) REFERENCES accounts(account_id)
    )''')
    # Payment amounts mostly match invoice amounts
    # T016: 230.00 vs INV-016 210.00 -> TRUE ERROR (+20)
    # T019: 200.00 vs INV-019 150.00 -> TRUE ERROR (+50)
    # T020: 300.50 vs INV-020 300.00 -> INTENTIONAL ($0.50 FX rounding)
    transactions = [
        ('T001', 'INV-001', 'ACC003', 100.00, '2023-02-03', 'payment', 'Order O001 payment'),
        ('T002', 'INV-002', 'ACC003', 200.00, '2023-02-05', 'payment', 'Order O002 payment'),
        ('T003', 'INV-003', 'ACC003', 150.00, '2023-02-07', 'payment', 'Order O003 payment'),
        ('T004', 'INV-004', 'ACC003', 240.00, '2023-02-10', 'payment', 'Order O004 payment'),
        ('T005', 'INV-005', 'ACC003', 160.00, '2023-02-12', 'payment', 'Order O005 payment'),
        ('T006', 'INV-006', 'ACC003', 180.00, '2023-02-14', 'payment', 'Order O006 payment'),
        ('T007', 'INV-007', 'ACC003', 236.00, '2023-02-17', 'payment', 'Order O007 payment'),
        ('T008', 'INV-008', 'ACC003', 150.00, '2023-02-20', 'payment', 'Order O008 payment'),
        ('T009', 'INV-009', 'ACC003', 325.00, '2023-02-22', 'payment', 'Order O009 payment'),
        ('T010', 'INV-010', 'ACC003', 165.00, '2023-02-24', 'payment', 'Order O010 payment'),
        ('T011', 'INV-011', 'ACC003', 230.00, '2023-02-27', 'payment', 'Order O011 payment'),
        ('T012', 'INV-012', 'ACC003', 210.00, '2023-03-02', 'payment', 'Order O012 payment'),
        ('T013', 'INV-013', 'ACC003', 280.00, '2023-03-04', 'payment', 'Order O013 payment'),
        ('T014', 'INV-014', 'ACC003', 155.00, '2023-03-07', 'payment', 'Order O014 payment'),
        ('T015', 'INV-015', 'ACC003', 350.00, '2023-03-10', 'payment', 'Order O015 payment'),
        ('T016', 'INV-016', 'ACC003', 230.00, '2023-03-12', 'payment', 'Order O016 payment'),   # ERROR: should be 210
        ('T017', 'INV-017', 'ACC003', 220.00, '2023-03-14', 'payment', 'Order O017 payment'),
        ('T018', 'INV-018', 'ACC003', 605.00, '2023-03-17', 'payment', 'Order O018 payment'),
        ('T019', 'INV-019', 'ACC003', 200.00, '2023-03-20', 'payment', 'Order O019 payment'),   # ERROR: should be 150
        ('T020', 'INV-020', 'ACC003', 300.50, '2023-03-22', 'payment', 'Order O020 payment'),   # INTENTIONAL: FX rounding
        ('T021', 'INV-021', 'ACC003', 200.00, '2023-03-26', 'payment', 'Order O099 payment'),
        ('T022', 'INV-022', 'ACC003', 350.00, '2023-03-29', 'payment', 'Order O098 payment'),
        # Non-order transactions
        ('T023', None, 'ACC005', 500.00, '2023-03-01', 'expense', 'Office rent Q1'),
        ('T024', None, 'ACC005', 1200.00, '2023-03-15', 'expense', 'Contractor services'),
        ('T025', None, 'ACC004', -75.00, '2023-03-20', 'refund', 'Customer return - order O005'),
    ]
    c.executemany('INSERT INTO transactions VALUES (?,?,?,?,?,?,?)', transactions)

    conn.commit()
    conn.close()


if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    create_sales_db()
    create_finance_db()
    print("Databases created successfully.")
