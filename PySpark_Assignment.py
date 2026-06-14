# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
spark = SparkSession.builder.appName("PySparkAssignment").getOrCreate()

# COMMAND ----------

orders_data = [
    (101, 1, "Laptop", "Electronics", 50000, 1, "2025-01-10", "Hyderabad"),
    (102, 2, "Mobile", "Electronics", 30000, 2, "2025-01-11", "Bangalore"),
    (103, 3, "TV", "Electronics", 45000, 1, "2025-01-12", "Chennai"),
    (104, 1, "Shoes", "Fashion", 3000, 2, "2025-01-13", "Hyderabad"),
    (105, 4, "Watch", "Fashion", 5000, 1, "2025-01-14", "Mumbai"),
    (106, 5, "Refrigerator", "Appliances", 35000, 1, "2025-01-15", "Delhi"),
    (107, 2, "AC", "Appliances", 40000, 1, "2025-01-16", "Bangalore"),
    (108, 6, "Mobile", "Electronics", 25000, 3, "2025-01-17", "Pune"),
    (109, 3, "Shoes", "Fashion", 3500, 2, "2025-01-18", "Chennai"),
    (110, 7, "Laptop", "Electronics", 60000, 1, "2025-01-19", "Kolkata"),
    (111, 8, "Watch", "Fashion", 4500, 2, "2025-01-20", "Delhi"),
    (112, 4, "Mobile", "Electronics", 28000, 1, "2025-01-21", "Mumbai"),
    (113, 9, "TV", "Electronics", 55000, 1, "2025-01-22", "Hyderabad"),
    (114, 10, "AC", "Appliances", 42000, 1, "2025-01-23", "Pune"),
    (115, 5, "Shoes", "Fashion", 2500, 4, "2025-01-24", "Delhi")
]

orders_cols = ["order_id","customer_id","product",
               "category","price","quantity",
               "order_date","city"]

orders_df = spark.createDataFrame(orders_data, orders_cols)

# COMMAND ----------

orders_df.show()

# COMMAND ----------

orders_df.printSchema()

# COMMAND ----------

orders_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC **Section-1:Basic Transformations**

# COMMAND ----------

orders_df.select("order_id","product","price").show()

# COMMAND ----------

q2_df = orders_df.withColumnRenamed(
    "product",
    "product_name"
)

q2_df.printSchema()

# COMMAND ----------

orders_df.printSchema()

# COMMAND ----------

q3_df = orders_df.withColumn(
    "total_amount",
    col("price") * col("quantity")
)

q3_df.show()

# COMMAND ----------

q4_df=q3_df.withColumn(
    "gstamount",
    col("total_amount")*0.18)
q4_df.show()


# COMMAND ----------

q4_df = orders_df.withColumn(
    "discount",
    when(col("category") == "Electronics", 10)
    .when(col("category") == "Fashion", 5)
    .when(col("category") == "Appliances", 8)
    .otherwise(0)
).show()

# COMMAND ----------

q5_df = orders_df.withColumn(
    "order_date",
    to_date(col("order_date"))
)

q5_df.show()

# COMMAND ----------

q5_df.printSchema()

# COMMAND ----------

q5_df.select(
    year("order_date")
).show()

# COMMAND ----------

orders_df.filter(col("city")=="Hyderabad").show()

# COMMAND ----------

orders_df.filter(
    col("quantity") > 2
).show()

# COMMAND ----------

orders_df.filter(col("category") == "Electronics").show()

# COMMAND ----------

q9_df = orders_df.withColumn(
    "category_upper",
    upper(col("category"))
)

q9_df.show()

# COMMAND ----------

q10=orders_df.drop("city").show()

# COMMAND ----------

customers_data = [
    (1,"Sravan",28,"Hyderabad"),
    (2,"Rahul",30,"Bangalore"),
    (3,"Priya",27,"Chennai"),
    (4,"Sneha",35,"Mumbai"),
    (5,"Arjun",32,"Delhi"),
    (6,"Kiran",29,"Pune"),
    (7,"Amit",40,"Kolkata"),
    (8,"Meena",26,"Delhi"),
    (9,"Ravi",31,"Hyderabad"),
    (10,"Pooja",33,"Pune")
]

customers_cols = ["customer_id","customer_name","age","city"]

customers_df = spark.createDataFrame(customers_data, customers_cols)

customers_df.show()
customers_df.printSchema()
customers_df.describe().show()

# COMMAND ----------

customers_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC **Section-2:Filter Transformations (11-15)**

# COMMAND ----------

orders_enriched_df = orders_df.withColumn(
    "total_amount",
    col("price") * col("quantity")
)

# COMMAND ----------

q11_df = orders_enriched_df.filter(
    col("total_amount") > 40000
)

q11_df.show()

# COMMAND ----------

orders_enriched_df = orders_enriched_df.withColumn(
    "order_date",
    to_date(col("order_date"))
)

# COMMAND ----------

q12_df = orders_enriched_df.filter(
    col("order_date").between(
        "2025-01-15",
        "2025-01-22"
    )
)

q12_df.show()

# COMMAND ----------

q13_df = customers_df.filter(
    col("age") > 30
)

q13_df.show()

# COMMAND ----------

q14_df = orders_df.filter(
    col("product").startswith("M")
)

q14_df.show()

# COMMAND ----------

q15_df = orders_df.filter(
    col("city").isin(
        "Hyderabad",
        "Bangalore"
    )
)

q15_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Section 3: Aggregations (16-22)**

# COMMAND ----------

from pyspark.sql.functions import sum

orders_enriched_df.select(
    sum("total_amount").alias("total_sales")
).show()

# COMMAND ----------

orders_enriched_df.groupBy("category") \
    .agg(
        sum("quantity").alias("total_quantity")
    ) \
    .show()

# COMMAND ----------

from pyspark.sql.functions import avg

orders_enriched_df.groupBy("category") \
    .agg(
        avg("price").alias("avg_price")
    ) \
    .show()

# COMMAND ----------

orders_enriched_df.select(
    max("price")
).show()
orders_enriched_df.orderBy(
    col("price").desc()
).show(1)

# COMMAND ----------

orders_enriched_df.orderBy(
    col("price").asc()
).show(1)

# COMMAND ----------

from pyspark.sql.functions import count

orders_enriched_df.groupBy("city") \
    .agg(
        count("*").alias("total_orders")
    ) \
    .show()

# COMMAND ----------

orders_enriched_df.groupBy("category") \
    .agg(
        sum("total_amount").alias("revenue")
    ) \
    .orderBy(
        col("revenue").desc()
    ) \
    .show(1)

# COMMAND ----------

# MAGIC %md
# MAGIC **Section 4: GroupBy Transformations (23-28)**

# COMMAND ----------

orders_df.groupBy("category") \
    .count() \
    .show()

# COMMAND ----------

orders_df.groupBy("city") \
    .agg(
        avg("quantity").alias("avg_quantity")
    ) \
    .show()

# COMMAND ----------

orders_enriched_df.groupBy("city") \
    .agg(
        sum("total_amount").alias("revenue")
    ) \
    .show()

# COMMAND ----------

orders_enriched_df.groupBy("customer_id") \
    .agg(
        sum("total_amount").alias("customer_revenue")
    ) \
    .show()

# COMMAND ----------

orders_enriched_df.groupBy("customer_id") \
    .agg(
        sum("total_amount").alias("customer_revenue")
    ) \
    .orderBy(
        col("customer_revenue").desc()
    ) \
    .limit(3) \
    .show()

# COMMAND ----------

orders_enriched_df.groupBy("city") \
    .agg(
        sum("total_amount").alias("sales")
    ) \
    .orderBy(
        col("sales").desc()
    ) \
    .show(1)

# COMMAND ----------

# MAGIC %md
# MAGIC **Section 5: Join Transformations (29-34)**

# COMMAND ----------

joined_df = orders_enriched_df.join(
    customers_df,
    on="customer_id",
    how="inner"
)

joined_df.show()

# COMMAND ----------

orders_enriched_df.join(
    customers_df,
    "customer_id"
).select(
    "customer_name",
    "product"
).show()

# COMMAND ----------

joined_df.select(
    "customer_name",
    "product"
).show()

# COMMAND ----------

orders_enriched_df.join(
    customers_df,
    "customer_id"
).groupBy(
    "customer_name"
).agg(
    sum("total_amount").alias("total_spending")
).show()

# COMMAND ----------

orders_enriched_df.join(
    customers_df,
    "customer_id"
).groupBy(
    "customer_name"
).agg(
    sum("total_amount").alias("total_spending")
).orderBy(
    col("total_spending").desc()
).limit(1).show()

# COMMAND ----------

orders_enriched_df.groupBy(
    "city"
).agg(
    avg("total_amount").alias("avg_spending")
).show()

# COMMAND ----------

orders_enriched_df.join(
    customers_df,
    "customer_id"
).filter(
    col("category") == "Electronics"
).select(
    "customer_name"
).distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Section 6: Sorting & Distinct (35-38)**

# COMMAND ----------

orders_df.orderBy(
    col("price").desc()
).show()

# COMMAND ----------

orders_df.select(
    "category"
).distinct().show()

# COMMAND ----------

orders_df.select("city").distinct().show()

# COMMAND ----------

orders_df.orderBy(col("price").desc()).show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC **Section 7: Window Functions (39-45)**
# MAGIC

# COMMAND ----------

#window
revenue_window = Window.orderBy(
    col("total_amount").desc()
)
orders_enriched_df.withColumn(
    "row_num",
    row_number().over(revenue_window)
).show()

# COMMAND ----------

#window
category_window = Window.partitionBy(
    "category"
).orderBy(
    col("price").desc()
)
orders_df.withColumn(
    "rank",
    rank().over(category_window)
).show()


# COMMAND ----------

orders_df.withColumn(
    "dense_rank",
    dense_rank().over(category_window)
).show()

# COMMAND ----------

#Find Highest Priced Product In Each Category
highest_df = orders_df.withColumn(
    "rank",
    rank().over(category_window)
)
highest_df.filter(
    col("rank") == 1
).show()


# COMMAND ----------

highest_df.filter(
    col("rank") == 2
).show()


# COMMAND ----------

#Calculate Running Revenue Per City
#window
city_window = Window.partitionBy(
    "city"
).orderBy(
    "order_date"
)
orders_enriched_df.withColumn(
    "running_revenue",
    sum("total_amount").over(city_window)
).show()

# COMMAND ----------

# Calculate Cumulative Quantity Sold
# Window
quantity_window = Window.orderBy(
    "order_date"
)
orders_df.withColumn(
    "cumulative_quantity",
    sum("quantity").over(quantity_window)
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Section 8: Advanced Transformations (46-55)**

# COMMAND ----------

# Classify Orders
# Requirement
# High Value   > 50000
# Medium Value 20000 - 50000
# Low Value    < 20000
# Solution
orders_enriched_df.withColumn(
    "value_category",
    when(col("total_amount") > 50000, "High Value")
    .when(
        (col("total_amount") >= 20000) &
        (col("total_amount") <= 50000),
        "Medium Value"
    )
    .otherwise("Low Value")
).show()

# COMMAND ----------

# Remove Duplicate Products
orders_df.dropDuplicates(
    ["product"]
).show()

# COMMAND ----------

# Pivot Category-Wise Sales By City
# Requirement
# Convert:
# City      Category      Revenue
# into:
# City   Electronics Fashion Appliances
# Solution
pivot_df = orders_enriched_df.groupBy("city") \
    .pivot("category") \
    .agg(sum("total_amount"))
pivot_df.show()

# COMMAND ----------

print(type(pivot_df))

# COMMAND ----------

#unpivot the pivoted data
unpivot_df = pivot_df.selectExpr(
    "city",
    "stack(3, \
    'Electronics', Electronics, \
    'Fashion', Fashion, \
    'Appliances', Appliances) \
    as (category,revenue)"
)

unpivot_df.show()

# COMMAND ----------

#Create Array Column
array_df = orders_df.withColumn(
    "product_info",
    array(
        col("product"),
        col("category")
    )
).show(truncate=False)

# COMMAND ----------

# Explode Array Column

array_df = orders_df.withColumn(
    "product_info",
    array(
        col("product"),
        col("category")
    )
)
array_df.select(
    "order_id",
    explode("product_info")
).show()

# COMMAND ----------

#create map column
orders_df.withColumn(
    "product_map",
    create_map(
        lit("Product"),
        col("product"),
        lit("Category"),
        col("category")
    )
).show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC **Section 9: Real-Time Data Engineer Scenarios (56-65)**

# COMMAND ----------

#Find Customers Purchasing From Multiple Categories
orders_df.join(
    customers_df,
    "customer_id"
).groupBy(
    "customer_name"
).agg(
    countDistinct("category").alias("category_count")
).filter(
    col("category_count") > 1
).show()

# COMMAND ----------

# Identify Customers Who Bought Both Mobile And Laptop
# Solution
orders_df.filter(
    col("product").isin("Mobile", "Laptop")
).groupBy(
    "customer_id"
).agg(
    countDistinct("product").alias("product_count")
).filter(
    col("product_count") == 2
).show()


# COMMAND ----------

# Find Repeat Customers
orders_df.join(
    customers_df,
    "customer_id"
).groupBy(
    "customer_name"
).agg(
    count("order_id").alias("order_count")
).filter(
    col("order_count") > 1
).show()

# COMMAND ----------

# Category Contribution Percentage To Total Revenue
category_revenue = orders_enriched_df.groupBy(
    "category"
).agg(
    sum("total_amount").alias("revenue")
)
total_revenue = orders_enriched_df.agg(
    sum("total_amount")
).collect()[0][0]
category_revenue.withColumn(
    "contribution_pct",
    round(
        col("revenue") * 100 / total_revenue,
        2
    )
).show()

# COMMAND ----------

# Monthly Sales Trend
# Convert Date
orders_enriched_df = orders_enriched_df.withColumn(
    "order_date",
    to_date("order_date")
)
# Solution
orders_enriched_df.groupBy(
    month("order_date").alias("month")
).agg(
    sum("total_amount").alias("monthly_sales")
).orderBy(
    "month"
).show()

# COMMAND ----------

orders_df.groupBy(
    "product"
).agg(
    sum("quantity").alias("total_qty")
).orderBy(
    col("total_qty").desc()
).show(1)

# COMMAND ----------

orders_df.groupBy(
    "product"
).agg(
    sum("quantity").alias("total_qty")
).orderBy(
    col("total_qty").asc()
).show(1)

# COMMAND ----------

# Customer Retention Percentage
# Simple Definition
# Repeat Customers / Total Customers
# Step 1
# Repeat Customers
repeat_customers = orders_df.groupBy(
    "customer_id"
).count().filter(
    col("count") > 1
).count()
# Step 2
# Total Customers
total_customers = orders_df.select(
    "customer_id"
).distinct().count()
# Step 3
# Retention %
retention_pct = (
    repeat_customers * 100
) / total_customers

print(retention_pct)

# COMMAND ----------

# City-Wise Revenue Contribution
# Revenue By City
city_revenue = orders_enriched_df.groupBy(
    "city"
).agg(
    sum("total_amount").alias("revenue")
)
# Total Revenue
total_revenue = orders_enriched_df.agg(
    sum("total_amount")
).collect()[0][0]
# Contribution %
city_revenue.withColumn(
    "contribution_pct",
    round(
        col("revenue") * 100 / total_revenue,
        2
    )
).show()

# COMMAND ----------

# Generate Customer Segmentation Report
# Business Rule
# Premium > 100000
# Gold    50000 - 100000
# Silver  < 50000
# Step 1
# Revenue Per Customer

customer_revenue = orders_enriched_df.groupBy(
    "customer_id"
).agg(
    sum("total_amount").alias("revenue")
)
# Step 2
# Segment
customer_revenue.withColumn(
    "segment",
    when(
        col("revenue") > 100000,
        "Premium"
    )
    .when(
        col("revenue") >= 50000,
        "Gold"
    )
    .otherwise("Silver")
).show()

# COMMAND ----------

# MAGIC %md
# MAGIC **Bonus Interview Questions (66-75)**

# COMMAND ----------

Bonus Interview Questions (66-75)
66.

Difference between select() and withColumn()?

67.

Difference between distinct() and dropDuplicates()?

68.

Difference between groupBy() and window functions()?

69.

When would you use broadcast join?

70.

Difference between repartition() and coalesce()?

71.

What causes data skew?

72.

How would you optimize a large join?

73.

How do you handle small files problem?

74.

Difference between narrow and wide transformations?

75.

How does Catalyst Optimizer improve performance?