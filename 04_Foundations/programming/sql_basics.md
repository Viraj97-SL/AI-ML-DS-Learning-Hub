# SQL Fundamentals for Data Science

> SQL is the lingua franca of data. Every DS, MLE, and AIE role expects proficiency. This guide takes you from basics to window functions and interview-level challenges — with self-contained, runnable examples throughout.

---

## Table of Contents
1. [Setup — SQLite in Python](#1-setup--sqlite-in-python)
2. [SELECT — The Foundation](#2-select--the-foundation)
3. [Filtering with WHERE](#3-filtering-with-where)
4. [Aggregations & GROUP BY](#4-aggregations--group-by)
5. [JOINs](#5-joins)
6. [Subqueries](#6-subqueries)
7. [Common Table Expressions (CTEs)](#7-common-table-expressions-ctes)
8. [Window Functions](#8-window-functions)
9. [String & Date Functions](#9-string--date-functions)
10. [Interview Challenges](#10-interview-challenges)

---

## 1. Setup — SQLite in Python

No server required. SQLite is built into Python.

```python
import sqlite3
import pandas as pd

# Connect (creates file if not exists; use :memory: for in-memory)
conn = sqlite3.connect(":memory:")
cursor = conn.cursor()

# Helper: run SQL and return a DataFrame
def sql(query, conn=conn):
    return pd.read_sql_query(query, conn)
```

### Create Sample Tables

```sql
-- Employees table
CREATE TABLE employees (
    emp_id     INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    dept       TEXT NOT NULL,
    salary     REAL NOT NULL,
    manager_id INTEGER,
    hire_date  TEXT   -- YYYY-MM-DD
);

-- Departments table
CREATE TABLE departments (
    dept_id   INTEGER PRIMARY KEY,
    dept_name TEXT NOT NULL,
    budget    REAL
);

-- Sales table
CREATE TABLE sales (
    sale_id   INTEGER PRIMARY KEY,
    emp_id    INTEGER,
    sale_date TEXT,
    amount    REAL,
    product   TEXT
);
```

```python
cursor.executescript("""
CREATE TABLE employees (
    emp_id     INTEGER PRIMARY KEY,
    name       TEXT,
    dept       TEXT,
    salary     REAL,
    manager_id INTEGER,
    hire_date  TEXT
);

CREATE TABLE departments (
    dept_id   INTEGER PRIMARY KEY,
    dept_name TEXT,
    budget    REAL
);

CREATE TABLE sales (
    sale_id   INTEGER PRIMARY KEY,
    emp_id    INTEGER,
    sale_date TEXT,
    amount    REAL,
    product   TEXT
);

INSERT INTO employees VALUES
    (1,  'Alice',   'Engineering', 120000, NULL, '2020-01-15'),
    (2,  'Bob',     'Engineering',  95000,    1, '2021-03-22'),
    (3,  'Carol',   'Engineering',  85000,    1, '2022-07-01'),
    (4,  'Dave',    'Sales',        75000, NULL, '2019-11-05'),
    (5,  'Eve',     'Sales',        68000,    4, '2021-09-12'),
    (6,  'Frank',   'Sales',        71000,    4, '2020-06-30'),
    (7,  'Grace',   'HR',           80000, NULL, '2018-04-17'),
    (8,  'Henry',   'HR',           65000,    7, '2023-02-28'),
    (9,  'Ivy',     'Marketing',    90000, NULL, '2017-10-01'),
    (10, 'Jack',    'Marketing',    78000,    9, '2022-05-15');

INSERT INTO departments VALUES
    (1, 'Engineering', 500000),
    (2, 'Sales',       300000),
    (3, 'HR',          200000),
    (4, 'Marketing',   250000),
    (5, 'Legal',       150000);

INSERT INTO sales VALUES
    (1,  1, '2024-01-10', 15000, 'Software'),
    (2,  2, '2024-01-15', 12000, 'Software'),
    (3,  4, '2024-01-20', 25000, 'Hardware'),
    (4,  5, '2024-02-05',  8000, 'Software'),
    (5,  4, '2024-02-10', 30000, 'Hardware'),
    (6,  6, '2024-02-15', 18000, 'Services'),
    (7,  1, '2024-03-01', 22000, 'Software'),
    (8,  5, '2024-03-05', 11000, 'Services'),
    (9,  4, '2024-03-20', 35000, 'Hardware'),
    (10, 6, '2024-04-01', 14000, 'Software');

conn.commit()
""")
```

---

## 2. SELECT — The Foundation

```sql
-- Select all columns
SELECT * FROM employees;

-- Select specific columns
SELECT name, dept, salary FROM employees;

-- Aliases
SELECT name AS employee_name,
       salary / 12 AS monthly_salary
FROM employees;

-- Distinct values
SELECT DISTINCT dept FROM employees;

-- Ordering results
SELECT name, salary
FROM employees
ORDER BY salary DESC;       -- highest first

-- Limiting rows
SELECT name, salary
FROM employees
ORDER BY salary DESC
LIMIT 3;                    -- top 3 earners

-- OFFSET for pagination
SELECT name, salary
FROM employees
ORDER BY salary DESC
LIMIT 3 OFFSET 3;           -- rows 4-6
```

**Execution order** (not how you write it, but how SQL processes it):
```
FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
```

---

## 3. Filtering with WHERE

```sql
-- Comparison operators
SELECT * FROM employees WHERE salary > 90000;
SELECT * FROM employees WHERE dept = 'Engineering';
SELECT * FROM employees WHERE hire_date >= '2022-01-01';

-- Logical operators
SELECT * FROM employees
WHERE dept = 'Engineering' AND salary > 90000;

SELECT * FROM employees
WHERE dept = 'Sales' OR dept = 'Marketing';

SELECT * FROM employees
WHERE NOT dept = 'HR';

-- IN — match any value in a list
SELECT * FROM employees
WHERE dept IN ('Engineering', 'Marketing');

-- BETWEEN — inclusive on both ends
SELECT * FROM employees
WHERE salary BETWEEN 70000 AND 95000;

-- LIKE — pattern matching
-- % matches any sequence, _ matches one character
SELECT * FROM employees WHERE name LIKE 'A%';      -- starts with A
SELECT * FROM employees WHERE name LIKE '%e';      -- ends with e
SELECT * FROM employees WHERE name LIKE '_o%';     -- second char is o

-- IS NULL / IS NOT NULL
SELECT * FROM employees WHERE manager_id IS NULL;   -- top-level managers
SELECT * FROM employees WHERE manager_id IS NOT NULL;

-- CASE — conditional logic
SELECT name,
       salary,
       CASE
           WHEN salary >= 100000 THEN 'Senior'
           WHEN salary >= 75000  THEN 'Mid'
           ELSE                       'Junior'
       END AS level
FROM employees;
```

---

## 4. Aggregations & GROUP BY

```sql
-- Aggregate functions
SELECT
    COUNT(*)           AS total_employees,
    COUNT(manager_id)  AS employees_with_manager,  -- NULLs not counted
    AVG(salary)        AS avg_salary,
    MAX(salary)        AS max_salary,
    MIN(salary)        AS min_salary,
    SUM(salary)        AS total_payroll
FROM employees;

-- GROUP BY — aggregate per group
SELECT dept,
       COUNT(*)    AS headcount,
       AVG(salary) AS avg_salary,
       MAX(salary) AS top_salary
FROM employees
GROUP BY dept
ORDER BY avg_salary DESC;

-- HAVING — filter on aggregated values (WHERE can't use aggregates)
SELECT dept,
       COUNT(*) AS headcount,
       AVG(salary) AS avg_salary
FROM employees
GROUP BY dept
HAVING AVG(salary) > 80000;   -- only depts with avg salary > 80k

-- GROUP BY multiple columns
SELECT dept, hire_date,
       COUNT(*) AS hired_count
FROM employees
GROUP BY dept, hire_date;

-- Grouping with CASE
SELECT
    CASE
        WHEN salary >= 100000 THEN 'Senior'
        WHEN salary >= 75000  THEN 'Mid'
        ELSE 'Junior'
    END AS band,
    COUNT(*)    AS count,
    AVG(salary) AS avg_salary
FROM employees
GROUP BY band
ORDER BY avg_salary DESC;
```

---

## 5. JOINs

### Visual Reference (ASCII Venn Diagrams)

```
INNER JOIN               LEFT JOIN                RIGHT JOIN
   A ∩ B                 A + (A∩B)               B + (A∩B)
  ┌──┬──┐               ┌──┬──┐                 ┌──┬──┐
  │  │██│               │██│██│                 │  │██│
  │  │██│               │██│██│                 │  │██│
  └──┴──┘               └──┴──┘                 └──┴──┘
 only matches          all of A                all of B

FULL OUTER JOIN          CROSS JOIN
  A ∪ B                  every combination
  ┌──┬──┐
  │██│██│
  │██│██│
  └──┴──┘
all rows, both sides
```

### INNER JOIN — only matching rows

```sql
-- Employees joined with their sales
SELECT e.name, e.dept, s.sale_date, s.amount, s.product
FROM employees e
INNER JOIN sales s ON e.emp_id = s.emp_id
ORDER BY s.amount DESC;
```

### LEFT JOIN — all from left, matching from right

```sql
-- All employees, including those with no sales (NULLs for sales columns)
SELECT e.name, e.dept, COALESCE(SUM(s.amount), 0) AS total_sales
FROM employees e
LEFT JOIN sales s ON e.emp_id = s.emp_id
GROUP BY e.emp_id, e.name, e.dept
ORDER BY total_sales DESC;
```

### FULL OUTER JOIN — all rows from both sides

```sql
-- All employees + all departments (SQLite uses UNION to simulate FULL OUTER)
SELECT e.name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept = d.dept_name

UNION

SELECT e.name, d.dept_name
FROM employees e
RIGHT JOIN departments d ON e.dept = d.dept_name;
-- Note: SQLite doesn't support RIGHT JOIN directly; use LEFT + UNION
```

### SELF JOIN — join a table to itself

```sql
-- Show each employee with their manager's name
SELECT e.name AS employee,
       m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.emp_id
ORDER BY manager NULLS LAST;
```

### Multi-table JOIN

```sql
-- Combine employees, their sales, and department budget
SELECT
    e.name,
    e.dept,
    d.budget AS dept_budget,
    COALESCE(SUM(s.amount), 0) AS total_sales,
    ROUND(COALESCE(SUM(s.amount), 0) / d.budget * 100, 1) AS pct_of_budget
FROM employees e
LEFT JOIN sales s      ON e.emp_id    = s.emp_id
LEFT JOIN departments d ON e.dept     = d.dept_name
GROUP BY e.emp_id, e.name, e.dept, d.budget
ORDER BY total_sales DESC;
```

---

## 6. Subqueries

### Subquery in WHERE

```sql
-- Employees earning more than the average salary
SELECT name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees)
ORDER BY salary DESC;

-- Employees who made at least one sale
SELECT name, dept
FROM employees
WHERE emp_id IN (SELECT DISTINCT emp_id FROM sales);

-- Employees who made NO sales (anti-join pattern)
SELECT name, dept
FROM employees
WHERE emp_id NOT IN (SELECT DISTINCT emp_id FROM sales);
```

### Correlated Subquery

```sql
-- For each employee, find their salary vs department average
SELECT name,
       dept,
       salary,
       ROUND((SELECT AVG(e2.salary)
              FROM employees e2
              WHERE e2.dept = e1.dept), 0) AS dept_avg,
       salary - ROUND((SELECT AVG(e2.salary)
                       FROM employees e2
                       WHERE e2.dept = e1.dept), 0) AS vs_avg
FROM employees e1
ORDER BY dept, salary DESC;
```

### Subquery in FROM (derived table)

```sql
-- Department summary used as a derived table
SELECT dept_summary.dept,
       dept_summary.avg_salary,
       dept_summary.headcount
FROM (
    SELECT dept,
           AVG(salary) AS avg_salary,
           COUNT(*)    AS headcount
    FROM employees
    GROUP BY dept
) AS dept_summary
WHERE dept_summary.headcount >= 2
ORDER BY dept_summary.avg_salary DESC;
```

---

## 7. Common Table Expressions (CTEs)

CTEs make complex queries readable by naming intermediate results.

```sql
-- Basic CTE
WITH dept_stats AS (
    SELECT dept,
           AVG(salary) AS avg_salary,
           COUNT(*)    AS headcount
    FROM employees
    GROUP BY dept
)
SELECT e.name, e.salary, d.avg_salary,
       ROUND(e.salary - d.avg_salary, 0) AS vs_avg
FROM employees e
JOIN dept_stats d ON e.dept = d.dept
ORDER BY vs_avg DESC;
```

### Chaining Multiple CTEs

```sql
-- Chain: sales summary → top performers → final enrichment
WITH
sales_by_emp AS (
    SELECT emp_id,
           SUM(amount)  AS total_sales,
           COUNT(*)     AS num_deals
    FROM sales
    GROUP BY emp_id
),
top_performers AS (
    SELECT emp_id, total_sales, num_deals
    FROM sales_by_emp
    WHERE total_sales > (SELECT AVG(total_sales) FROM sales_by_emp)
)
SELECT e.name, e.dept, t.total_sales, t.num_deals
FROM employees e
JOIN top_performers t ON e.emp_id = t.emp_id
ORDER BY t.total_sales DESC;
```

### Recursive CTE — Org Chart / Hierarchy

```sql
-- Walk the org hierarchy from top to bottom
WITH RECURSIVE org_tree AS (
    -- Base case: top-level managers (no manager)
    SELECT emp_id, name, dept, manager_id, 0 AS level, name AS path
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case: employees reporting to someone in org_tree
    SELECT e.emp_id, e.name, e.dept, e.manager_id,
           o.level + 1,
           o.path || ' → ' || e.name
    FROM employees e
    JOIN org_tree o ON e.manager_id = o.emp_id
)
SELECT level, path, dept
FROM org_tree
ORDER BY path;
```

---

## 8. Window Functions

Window functions operate **across a set of rows related to the current row** without collapsing them like GROUP BY does.

```sql
-- Syntax
function_name() OVER (
    PARTITION BY column     -- optional: reset per group
    ORDER BY column         -- optional: define row order within window
    ROWS/RANGE frame_spec   -- optional: define the "window frame"
)
```

### Ranking Functions

```sql
SELECT name, dept, salary,
    ROW_NUMBER()   OVER (PARTITION BY dept ORDER BY salary DESC) AS row_num,
    RANK()         OVER (PARTITION BY dept ORDER BY salary DESC) AS rank,
    DENSE_RANK()   OVER (PARTITION BY dept ORDER BY salary DESC) AS dense_rank,
    NTILE(4)       OVER (ORDER BY salary DESC)                   AS quartile
FROM employees
ORDER BY dept, salary DESC;

-- Difference between RANK() and DENSE_RANK():
-- If two people tie at rank 1, RANK gives next person rank 3,
-- DENSE_RANK gives next person rank 2 (no gaps)
```

### LAG and LEAD — Access Adjacent Rows

```sql
-- Month-over-month sales comparison
WITH monthly AS (
    SELECT
        STRFTIME('%Y-%m', sale_date) AS month,
        SUM(amount) AS revenue
    FROM sales
    GROUP BY month
)
SELECT month,
       revenue,
       LAG(revenue, 1)  OVER (ORDER BY month) AS prev_month,
       LEAD(revenue, 1) OVER (ORDER BY month) AS next_month,
       ROUND(
           (revenue - LAG(revenue, 1) OVER (ORDER BY month))
           / LAG(revenue, 1) OVER (ORDER BY month) * 100, 1
       ) AS mom_growth_pct
FROM monthly;
```

### Running Totals and Moving Averages

```sql
SELECT name, dept, salary,
    -- Running total within each dept (ordered by hire_date)
    SUM(salary)  OVER (PARTITION BY dept ORDER BY hire_date
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                 AS running_total,

    -- 3-row moving average (current + 2 preceding)
    AVG(salary)  OVER (ORDER BY hire_date
                       ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
                 AS moving_avg_3,

    -- Salary as percent of department total
    ROUND(salary / SUM(salary) OVER (PARTITION BY dept) * 100, 1)
                 AS pct_of_dept
FROM employees
ORDER BY dept, hire_date;
```

### FIRST_VALUE / LAST_VALUE

```sql
SELECT name, dept, salary,
    FIRST_VALUE(name)   OVER (PARTITION BY dept ORDER BY salary DESC) AS top_earner,
    LAST_VALUE(salary)  OVER (
        PARTITION BY dept ORDER BY salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS lowest_in_dept
FROM employees;
```

---

## 9. String & Date Functions

```sql
-- String functions
SELECT
    UPPER(name)                   AS upper_name,
    LOWER(dept)                   AS lower_dept,
    LENGTH(name)                  AS name_length,
    SUBSTR(name, 1, 3)            AS first_3_chars,
    name || ' (' || dept || ')'   AS name_dept,
    TRIM('  hello  ')             AS trimmed,
    REPLACE(dept, 'Engineering', 'Eng') AS short_dept
FROM employees;

-- NULL handling
SELECT
    COALESCE(manager_id, -1)       AS mgr_or_minus1,  -- first non-NULL
    NULLIF(dept, 'HR')             AS dept_no_hr      -- returns NULL if equal
FROM employees;

-- Date functions (SQLite)
SELECT
    hire_date,
    STRFTIME('%Y', hire_date)     AS year,
    STRFTIME('%m', hire_date)     AS month,
    STRFTIME('%Y-%m', hire_date)  AS year_month,
    -- Days since hire
    CAST(
        (JULIANDAY('now') - JULIANDAY(hire_date))
    AS INTEGER)                   AS days_employed,
    -- Add 90 days
    DATE(hire_date, '+90 days')   AS after_probation
FROM employees;
```

---

## 10. Interview Challenges

### 🟢 Easy

**E1. Second Highest Salary**
```sql
-- Method 1: OFFSET
SELECT salary AS SecondHighestSalary
FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET 1;

-- Method 2: Subquery (handles NULL if fewer than 2 rows)
SELECT MAX(salary) AS SecondHighestSalary
FROM employees
WHERE salary < (SELECT MAX(salary) FROM employees);
```

**E2. Employees With Salary Above Department Average**
```sql
WITH dept_avg AS (
    SELECT dept, AVG(salary) AS avg_sal
    FROM employees GROUP BY dept
)
SELECT e.name, e.dept, e.salary, ROUND(d.avg_sal) AS dept_avg
FROM employees e
JOIN dept_avg d ON e.dept = d.dept
WHERE e.salary > d.avg_sal
ORDER BY e.dept, e.salary DESC;
```

**E3. Count Employees Per Department Hired Each Year**
```sql
SELECT dept,
       STRFTIME('%Y', hire_date) AS year,
       COUNT(*) AS hires
FROM employees
GROUP BY dept, year
ORDER BY dept, year;
```

---

### 🟡 Medium

**M1. Top Earner Per Department (No LIMIT per group)**
```sql
-- Method 1: Window function (preferred)
WITH ranked AS (
    SELECT name, dept, salary,
           RANK() OVER (PARTITION BY dept ORDER BY salary DESC) AS rnk
    FROM employees
)
SELECT name, dept, salary
FROM ranked
WHERE rnk = 1;

-- Method 2: Correlated subquery
SELECT name, dept, salary
FROM employees e
WHERE salary = (
    SELECT MAX(salary) FROM employees e2 WHERE e2.dept = e.dept
);
```

**M2. Running 7-Day Revenue (Moving Sum)**
```sql
WITH daily AS (
    SELECT sale_date, SUM(amount) AS daily_rev
    FROM sales
    GROUP BY sale_date
)
SELECT sale_date,
       daily_rev,
       SUM(daily_rev) OVER (
           ORDER BY sale_date
           ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
       ) AS rolling_7d_revenue
FROM daily
ORDER BY sale_date;
```

**M3. Month-over-Month Revenue Growth %**
```sql
WITH monthly AS (
    SELECT STRFTIME('%Y-%m', sale_date) AS month,
           SUM(amount) AS revenue
    FROM sales
    GROUP BY month
)
SELECT month,
       revenue,
       LAG(revenue) OVER (ORDER BY month) AS prev_revenue,
       ROUND(
           (revenue - LAG(revenue) OVER (ORDER BY month)) * 100.0
           / LAG(revenue) OVER (ORDER BY month), 1
       ) AS growth_pct
FROM monthly;
```

**M4. De-duplicate — Keep Latest Record Per Employee**
```sql
-- Imagine a log table with duplicate employee records
-- Keep only the row with the highest sale_id per emp_id
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY emp_id ORDER BY sale_id DESC) AS rn
    FROM sales
)
SELECT sale_id, emp_id, sale_date, amount, product
FROM ranked
WHERE rn = 1;
```

**M5. Cumulative % of Total Sales**
```sql
WITH emp_sales AS (
    SELECT emp_id, SUM(amount) AS total
    FROM sales GROUP BY emp_id
),
sorted AS (
    SELECT emp_id, total,
           SUM(total) OVER (ORDER BY total DESC
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative,
           SUM(total) OVER () AS grand_total
    FROM emp_sales
)
SELECT e.name,
       s.total,
       s.cumulative,
       ROUND(s.cumulative / s.grand_total * 100, 1) AS cum_pct
FROM sorted s
JOIN employees e ON s.emp_id = e.emp_id
ORDER BY s.total DESC;
```

**M6. Pivot — Revenue by Product per Month**
```sql
-- SQLite CASE-based pivot (no PIVOT keyword in SQLite)
SELECT STRFTIME('%Y-%m', sale_date) AS month,
       SUM(CASE WHEN product = 'Software' THEN amount ELSE 0 END) AS software,
       SUM(CASE WHEN product = 'Hardware' THEN amount ELSE 0 END) AS hardware,
       SUM(CASE WHEN product = 'Services' THEN amount ELSE 0 END) AS services
FROM sales
GROUP BY month
ORDER BY month;
```

**M7. Consecutive Dates Without a Gap**
```sql
-- Find runs of consecutive sale dates (islands and gaps)
WITH dated AS (
    SELECT DISTINCT sale_date,
           JULIANDAY(sale_date) - ROW_NUMBER() OVER (ORDER BY sale_date) AS grp
    FROM sales
)
SELECT MIN(sale_date) AS streak_start,
       MAX(sale_date) AS streak_end,
       COUNT(*)       AS streak_length
FROM dated
GROUP BY grp
ORDER BY streak_start;
```

---

### 🔴 Hard

**H1. Median Salary (No built-in MEDIAN in SQLite)**
```sql
WITH ordered AS (
    SELECT salary,
           ROW_NUMBER() OVER (ORDER BY salary) AS rn,
           COUNT(*) OVER ()                    AS cnt
    FROM employees
)
SELECT AVG(salary) AS median_salary
FROM ordered
WHERE rn IN (
    (cnt + 1) / 2,      -- lower middle
    (cnt + 2) / 2       -- upper middle (same as lower if odd count)
);
```

**H2. Salary Self-Join — Find Pairs of Employees with Same Salary**
```sql
SELECT a.name AS emp1, b.name AS emp2, a.salary
FROM employees a
JOIN employees b ON a.salary = b.salary
                 AND a.emp_id < b.emp_id   -- avoid (A,B) and (B,A) duplicates
ORDER BY a.salary DESC;
```

**H3. Year-over-Year Sales Growth by Department**
```sql
WITH annual AS (
    SELECT e.dept,
           STRFTIME('%Y', s.sale_date) AS yr,
           SUM(s.amount) AS revenue
    FROM sales s
    JOIN employees e ON s.emp_id = e.emp_id
    GROUP BY e.dept, yr
)
SELECT dept, yr, revenue,
       LAG(revenue) OVER (PARTITION BY dept ORDER BY yr) AS prev_yr,
       ROUND(
           (revenue - LAG(revenue) OVER (PARTITION BY dept ORDER BY yr))
           * 100.0 / LAG(revenue) OVER (PARTITION BY dept ORDER BY yr), 1
       ) AS yoy_growth_pct
FROM annual
ORDER BY dept, yr;
```

**H4. nth Highest Salary (Parameterizable)**
```sql
-- Return the 3rd highest salary (change 3 to any N)
SELECT salary
FROM (
    SELECT DISTINCT salary,
           DENSE_RANK() OVER (ORDER BY salary DESC) AS rnk
    FROM employees
) ranked
WHERE rnk = 3;
```

**H5. Find Employees Who Never Made a Sale (EXISTS anti-pattern)**
```sql
SELECT name, dept
FROM employees e
WHERE NOT EXISTS (
    SELECT 1 FROM sales s WHERE s.emp_id = e.emp_id
);
```

---

## Using SQL with Pandas

```python
import pandas as pd
import sqlite3

conn = sqlite3.connect(":memory:")  # your connection

# Read entire table into DataFrame
df_employees = pd.read_sql("SELECT * FROM employees", conn)

# Run complex SQL, get DataFrame back
df_result = pd.read_sql("""
    WITH dept_stats AS (
        SELECT dept, AVG(salary) AS avg_salary, COUNT(*) AS headcount
        FROM employees GROUP BY dept
    )
    SELECT e.name, e.dept, e.salary, d.avg_salary
    FROM employees e
    JOIN dept_stats d ON e.dept = d.dept
    WHERE e.salary > d.avg_salary
    ORDER BY e.salary DESC
""", conn)

# Write DataFrame back to SQL
df_employees.to_sql("employees_backup", conn, if_exists="replace", index=False)

# DuckDB — best for analytics on DataFrames (faster than SQLite for analytics)
# pip install duckdb
import duckdb
result = duckdb.sql("SELECT dept, AVG(salary) FROM df_employees GROUP BY dept").df()
```

---

## Quick Reference Card

| Clause | Purpose | Can filter on aggregates? |
|--------|---------|--------------------------|
| WHERE | Filter rows before aggregation | No |
| HAVING | Filter groups after aggregation | Yes |
| GROUP BY | Collapse rows into groups | — |
| ORDER BY | Sort result | — |
| LIMIT / OFFSET | Restrict rows returned | — |

| Window Function | What it returns |
|----------------|----------------|
| ROW_NUMBER() | Unique sequential number (no ties) |
| RANK() | Rank with gaps on ties (1, 1, 3) |
| DENSE_RANK() | Rank without gaps on ties (1, 1, 2) |
| LAG(col, n) | Value from n rows before current |
| LEAD(col, n) | Value from n rows after current |
| FIRST_VALUE(col) | First value in the window frame |
| SUM() OVER | Running/cumulative sum |
| AVG() OVER | Moving/cumulative average |

---

*Back to: [Programming Foundations](.) | [Mathematics](../mathematics/) | [Main README](../../README.md)*
