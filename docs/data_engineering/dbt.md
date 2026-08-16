### Data Engineering

#### Overview

Data Engineering is the backbone of the data world. It focuses on designing, building, and maintaining the systems and infrastructure that allow for the large-scale collection, storage, and processing of data. A data engineer creates the "pipelines" that transport data from various sources to a central repository, like a data warehouse or data lake. They ensure this data is clean, reliable, and accessible so that data scientists and analysts can perform their analyses. Key skills include proficiency in programming (like Python), database technologies (SQL and NoSQL), and tools for data processing (like Spark and Airflow) and cloud computing.

---

#### Resources

A curated list of resources to help you get started and advance your data engineering skills.

##### Roadmaps & Overviews

*   [The Data Engineering Roadmap for Beginners](https://www.dataquest.io/blog/the-data-engineer-roadmap-for-beginners) - A comprehensive guide from Dataquest on what to learn and in what order.
*   [Start Data Engineering Blog](https://www.startdataengineering.com) - An excellent blog with articles on modern data engineering practices and interview preparation.
*   [How I would learn Data Engineering in 2026 (Video)](https://www.youtube.com/watch?v=iiV_O4Uqj9Q) - A practical video guide on building a learning plan from the ground up.
*   [The 2025 Data Engineering Roadmap (Video)](https://www.youtube.com/watch?v=aSHg22oEGIs) - A video outlining the key technologies and concepts for the coming year.
*   [A Non-Beginner Data Engineering Roadmap](https://blog.dataengineerthings.org/a-non-beginner-data-engineering-roadmap-2025-edition-2b39d865dd0b) - For when you've mastered the basics and are ready for the next level.

##### Courses & Tutorials

*   [Data Engineering Course for Beginners (YouTube)](https://www.youtube.com/watch?v=PHsC_t0j1dU) - A free, comprehensive video course covering databases, Docker, and analytical engineering.
*   [Fundamentals Of Data Engineering Masterclass (YouTube)](https://www.youtube.com/watch?v=hf2go3E2m8g) - A one-shot video covering the fundamental concepts of data engineering.
*   [Data Engineering Specialization by DeepLearning.AI](https://www.deeplearning.ai/specializations/data-engineering) - A structured specialization covering a wide range of data engineering topics.
*   [Data Engineering Academy](https://learndataengineering.com/p/academy) - In-depth courses on essential tools like Spark, Kafka, dbt, and Airflow.
*   [Learn Snowflake - Full 1-Hour Crash Course](https://www.youtube.com/watch?v=2t-ls6ekA8E) - A beginner-friendly introduction to the popular cloud data platform, Snowflake.

##### Interview Preparation

*   [Top 30 Data Engineer Interview Questions (Video)](https://www.youtube.com/watch?v=N-MbyH7EhoQ) - A video walkthrough of common interview questions and how to answer them.
*   [Scenario-Based PWC Data Engineering Interview Question (Video)](https://www.youtube.com/watch?v=2Hpb8ADLOI0) - Practice your problem-solving skills with a real-world scenario-based question.

##### Deeper Dives & Research

*   [Top 10 Data Engineering Research Papers to Read](https://dataheimer.substack.com/p/top-10-data-engineering-research) - A list of influential papers (like the one on Kafka) that shaped the field.
*   [One Week to Rebuild My Python Foundations as a Data Engineer](https://blog.devgenius.io/one-week-to-rebuild-my-python-foundations-as-a-data-engineer-30aa5cdc2369) - A practical guide to refreshing the core Python skills needed for data engineering.

---

#### Projects & Exercises

*   **Build a Personal ETL Pipeline:**
    1.  **Extract:** Choose a public API (e.g., OpenWeatherMap for weather data or a sports API for game stats). Write a Python script to fetch data from it.
    2.  **Transform:** Clean the data. You might convert temperatures from Kelvin to Celsius, select only the fields you need, or flatten a nested JSON structure.
    3.  **Load:** Load the cleaned data into a local SQLite or PostgreSQL database. Schedule your script to run once a day.

*   **Containerize a Data Processing Script:**
    1.  Write a simple Python script that reads a CSV file, performs a basic transformation (e.g., calculates a new column), and saves the result to a new CSV.
    2.  Write a `Dockerfile` to create an image that runs this script.
    3.  Build and run the Docker container to execute your script. This is a fundamental skill for creating reproducible data environments.

*   **Analyze Data with SQL:**
    1.  Download a public dataset from a source like Kaggle.
    2.  Use a database client (like DBeaver) or a command-line interface to create a table and load the dataset into a PostgreSQL database.
    3.  Write at least five different SQL queries to explore the data. Use `GROUP BY`, `JOIN`, and window functions to answer interesting questions about the dataset.
