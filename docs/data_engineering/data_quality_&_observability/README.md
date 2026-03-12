### Data Engineering

#### Overview

Data Engineering is the backbone of the data world. It focuses on designing, building, and maintaining the systems and infrastructure that allow for the large-scale collection, storage, and processing of data. While data scientists analyze data to find insights, data engineers create the robust "pipelines" that make this data available, reliable, and accessible. Key skills include programming (especially Python and SQL), understanding databases, building ETL/ELT processes, and working with cloud platforms and big data technologies.

#### Resources

##### Getting Started & Roadmaps

*   [Complete Guide: How to Become a Data Engineer in 2025](https://dataengineeracademy.com/blog/how-to-become-a-data-engineer/) - A detailed blog post outlining the steps and skills needed for the role.
*   [The 2025 data engineering roadmap](https://www.youtube.com/watch?v=aSHg22oEGIs) - A video guide to planning your learning journey.
*   [Data Engineering 2025: The Ultimate 1-Hour Crash Course](https://www.youtube.com/watch?v=CdyuRAzXbwA) - A quick, high-level introduction to the field.

##### Tutorials & Courses

*   [AWS Data Engineer Full Course in 10 Hours [2025]](https://www.youtube.com/watch?v=vQbReDn3GTs) - A comprehensive video course focused on data engineering within the AWS ecosystem.
*   [Data Engineering with Python | End-to-End Tutorial for 2025](https://www.youtube.com/watch?v=BgiOmkXgpno) - A practical, project-based tutorial using Python.
*   [Data Engineering for Machine Learning](https://madewithml.com/courses/mlops/data-engineering/) - A course that connects data engineering concepts directly to machine learning workflows.

##### Tool-Specific Guides

*   [Python Essentials for Data Engineers](https://www.startdataengineering.com/post/python-for-de/) - A guide to the core Python concepts most relevant to data engineering tasks.
*   [Intro to Data Engineering using Python in Snowflake](https://www.snowflake.com/en/developers/guides/intro-to-data-engineering-python/) - Learn how to apply Python skills within the Snowflake data platform.

#### Projects & Exercises

1.  **Build a Personal ETL Pipeline:**
    *   **Extract:** Choose a public API (e.g., a weather API, a stock market API, or a public transit API). Write a Python script using the `requests` library to fetch data from it.
    *   **Transform:** Using Python, clean the raw JSON data. Select only the fields you need, rename columns, and handle any missing values.
    *   **Load:** Load the cleaned data into a local SQLite database. Your script should create the table if it doesn't exist and append the new data each time it runs.

2.  **Design a Simple Data Warehouse Schema:**
    *   Imagine you are working for a small online bookstore.
    *   Design a "star schema" for their sales data. What would your central "fact" table be (e.g., `sales_transactions`)?
    *   What would your "dimension" tables be (e.g., `customers`, `books`, `dates`)?
    *   Write the SQL `CREATE TABLE` statements for this schema.

3.  **Automate a Data Collection Script:**
    *   Take the ETL pipeline you built in the first exercise.
    *   Use a simple scheduler to run it automatically. If you are on macOS or Linux, learn how to use `cron`. If you are on Windows, use the Task Scheduler.
    *   Set up your script to run once every day to collect new data.
