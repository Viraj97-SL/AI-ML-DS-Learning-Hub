# Data Quality & Observability

> A practitioner's guide to ensuring your data is trustworthy — covering validation frameworks, observability tooling, data contracts, and building quality into pipelines from day one.

---

## Overview

Data quality and observability are two sides of the same coin. **Data quality** is about ensuring data meets defined standards (correct, complete, consistent, timely). **Data observability** is about maintaining visibility into the health of your data systems in production — knowing *when* something breaks and *why*, even before downstream users notice.

Bad data is expensive. Studies consistently show that poor data quality costs organizations between 15–25% of revenue. A model trained on bad data will make bad predictions. A BI dashboard built on stale or duplicated records misleads decision-makers. Data teams that invest in quality and observability ship faster and with more confidence.

The modern data stack has shifted quality left — embedding validation into pipelines rather than catching problems after the fact. Tools like **Great Expectations**, **dbt tests**, and **Pandera** let you define expectations declaratively; observability platforms like **Monte Carlo** and **Bigeye** monitor for anomalies continuously.

---

## Key Concepts

### The 6 Dimensions of Data Quality

| Dimension | Definition | Example Check |
|-----------|-----------|---------------|
| **Completeness** | No missing values where required | `NOT NULL` on critical columns |
| **Accuracy** | Values are correct and truthful | Age between 0 and 120 |
| **Consistency** | No contradictions across sources | Same customer_id → same name everywhere |
| **Timeliness** | Data arrives when expected | Pipeline ran within last 25 hours |
| **Uniqueness** | No unexpected duplicates | Primary key has no duplicates |
| **Validity** | Values conform to a defined format | Email matches regex, dates are valid |

### Data Observability Pillars (Monte Carlo's 5 Pillars)
1. **Freshness** — Is data up to date? When was the table last updated?
2. **Volume** — Did the expected number of rows arrive?
3. **Schema** — Did columns change, get added, or get dropped?
4. **Distribution** — Are value distributions within expected ranges?
5. **Lineage** — Which upstream tables affect this table?

### Data Contracts
A **data contract** is a formal agreement between data producers and consumers that specifies:
- Schema (column names, types, nullability)
- Semantics (what each column means)
- SLAs (freshness, availability)
- Breaking change policy

Data contracts prevent "silent failures" where upstream changes break downstream models or reports without warning.

---

## Learning Path

### Beginner
1. Understand the 6 dimensions of data quality
2. Write basic SQL data quality checks (COUNT, NULL checks, range checks)
3. Learn Great Expectations basics — create your first expectation suite
4. Use dbt `tests` block for schema and referential integrity tests

### Intermediate
5. Build a data validation pipeline with Pandera for pandas DataFrames
6. Implement column-level data lineage with dbt `ref()` and `source()`
7. Set up anomaly detection alerts on table freshness and row counts
8. Write data contracts using YAML schemas
9. Implement custom Great Expectations validators

### Advanced
10. Design a data quality scoring framework (per-table quality scores)
11. Integrate observability into CI/CD (fail pipelines on quality regressions)
12. Implement schema registry and evolution strategies
13. Build SLA dashboards for data pipeline health
14. Implement data quarantine patterns for bad records

---

## Code Examples

### Great Expectations — Validation Suite

```python
import great_expectations as gx
import pandas as pd

# Load your DataFrame
df = pd.read_csv("customer_transactions.csv")

# Create a GX context
context = gx.get_context()

# Add a pandas datasource
datasource = context.sources.add_pandas("my_datasource")
asset = datasource.add_dataframe_asset("transactions")
batch_request = asset.build_batch_request(dataframe=df)

# Create expectation suite
suite = context.add_expectation_suite("transactions_suite")
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite=suite,
)

# Define expectations
validator.expect_column_to_exist("customer_id")
validator.expect_column_values_to_not_be_null("customer_id")
validator.expect_column_values_to_be_unique("transaction_id")
validator.expect_column_values_to_be_between("amount", min_value=0.01, max_value=100_000)
validator.expect_column_values_to_match_regex(
    "email",
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)
validator.expect_column_values_to_be_in_set(
    "status",
    ["pending", "completed", "refunded", "failed"]
)
validator.expect_table_row_count_to_be_between(min_value=1000, max_value=10_000_000)
validator.expect_column_values_to_not_be_null("amount", mostly=0.99)  # 99% non-null

# Save and run
validator.save_expectation_suite()
results = validator.validate()

print(f"Success: {results.success}")
print(f"Passed: {results.statistics['successful_expectations']} / "
      f"{results.statistics['evaluated_expectations']}")

# Show failures
for result in results.results:
    if not result.success:
        print(f"FAIL: {result.expectation_config.expectation_type} "
              f"on column '{result.expectation_config.kwargs.get('column', 'table')}'")
```

### Pandera — Schema Validation for pandas

```python
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import pandas as pd
from datetime import datetime

# Define schema declaratively
transaction_schema = DataFrameSchema(
    columns={
        "transaction_id": Column(
            pa.String,
            checks=Check.str_matches(r"^TXN-\d{10}$"),
            nullable=False,
            unique=True,
        ),
        "customer_id": Column(pa.Int64, nullable=False),
        "amount": Column(
            pa.Float64,
            checks=[
                Check.greater_than(0),
                Check.less_than_or_equal_to(100_000),
            ],
        ),
        "currency": Column(
            pa.String,
            checks=Check.isin(["USD", "EUR", "GBP", "JPY"]),
        ),
        "created_at": Column(pa.DateTime, nullable=False),
        "status": Column(
            pa.String,
            checks=Check.isin(["pending", "completed", "refunded", "failed"]),
        ),
    },
    checks=[
        # Table-level check: no duplicate (customer_id, created_at) pairs
        Check(lambda df: ~df.duplicated(subset=["customer_id", "created_at"]).any()),
    ],
    coerce=True,     # attempt type coercion before failing
    strict=False,    # allow extra columns not in schema
)

# Use as a decorator for pipeline functions
@pa.check_input(transaction_schema, "df")
@pa.check_output(transaction_schema)
def process_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Validated transformation — schema checked on input and output."""
    df["amount_usd"] = df["amount"]  # simplified
    return df

# Manual validation
try:
    validated_df = transaction_schema.validate(raw_df, lazy=True)
except pa.errors.SchemaErrors as e:
    print("Validation failures:")
    print(e.failure_cases)
```

### dbt Data Quality Tests (YAML)

```yaml
# models/schema.yml
version: 2

models:
  - name: orders
    description: "Cleaned orders table from raw e-commerce data"
    columns:
      - name: order_id
        description: "Primary key"
        tests:
          - unique
          - not_null

      - name: customer_id
        tests:
          - not_null
          - relationships:
              to: ref('customers')
              field: customer_id

      - name: status
        tests:
          - accepted_values:
              values: ['placed', 'shipped', 'delivered', 'cancelled']

      - name: amount
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: ">= 0"

    # Table-level test: freshness
    freshness:
      warn_after: {count: 12, period: hour}
      error_after: {count: 24, period: hour}
```

---

## Tools & Libraries

| Tool | Category | Description |
|------|----------|-------------|
| **Great Expectations** | Validation | Declarative expectation suites, data docs |
| **Pandera** | Validation | pandas/Polars schema validation, type safety |
| **dbt tests** | Pipeline quality | SQL-level tests baked into transformation |
| **Pydantic** | Schema validation | Python data models with type validation |
| **Monte Carlo** | Observability | ML-powered anomaly detection for data |
| **Bigeye** | Observability | Column-level monitoring and alerting |
| **Soda Core** | Validation/Monitoring | Open-source data monitoring framework |
| **Elementary** | dbt observability | Open-source dbt-native observability |
| **Datahub / OpenMetadata** | Data catalog + lineage | Metadata management, column lineage |
| **OpenLineage** | Lineage standard | Open standard for data lineage tracking |

---

## Resources

### Courses & Tutorials
- [Data Engineering Zoomcamp — Data Quality Module](https://github.com/DataTalksClub/data-engineering-zoomcamp) — Free, comprehensive DE course
- [Great Expectations Tutorials](https://docs.greatexpectations.io/docs/tutorials/) — Official GX documentation with hands-on examples
- [dbt Courses](https://courses.getdbt.com/) — Free dbt fundamentals and advanced courses

### Books
- *Fundamentals of Data Engineering* — Reis & Housley (O'Reilly, 2022) — Best modern DE book; covers quality throughout
- *Data Management at Scale* — Schroeder & Wiese — Enterprise data quality patterns

### Key Resources
- [The Data Contract Specification](https://datacontract.com/) — Open specification for data contracts
- [dbt Labs: Data Reliability Engineering](https://www.getdbt.com/blog/data-reliability-engineering) — Practical guide to making data trustworthy

---

## Projects & Exercises

**Project 1 — Validation Pipeline**
Take a messy public dataset (e.g., NYC taxi data or OpenFDA drug events). Define a Pandera schema with all expected constraints. Run it against 3 months of data. Document every quality issue found and write a brief data quality report.

**Project 2 — dbt Quality Layer**
Set up a dbt project on a local DuckDB database. Load raw data, write 3 staging models, add `not_null`, `unique`, `accepted_values`, and `relationships` tests to every model. Configure freshness checks. Run `dbt test` and fix all failures.

**Project 3 — Observability Dashboard**
Build a simple data observability system in Python: for each table in a SQLite/DuckDB database, log daily row counts, null rates for key columns, and schema snapshots. Alert (print/email) when a metric deviates >20% from its 7-day average. Visualize with Streamlit.

---

## Related Topics
- [Data Engineer Track Overview →](../../02_ML_Engineer/README.md)
- [Feature Stores →](../../02_ML_Engineer/intermediate/06_feature_stores.ipynb)
- [MLOps & Cloud Platforms →](../mlops_and_production_ml/cloud_ml_platforms/README.md)
