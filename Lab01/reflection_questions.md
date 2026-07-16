1. Why is it important to understand the dataset before building a Machine Learning model?
Understanding the dataset (Exploratory Data Analysis) is critical because models learn directly from the data you feed them—often summarized by the principle "garbage in, garbage out." Before modeling, you must identify data types, spot missing or anomalous values, understand class imbalances, and uncover correlations. This understanding dictates how you clean the data, which features you engineer, and which ML algorithms are mathematically appropriate for the problem.

2. Which table appears to be the central table in the Olist dataset? Why?
The olist_orders_dataset is the central table. It acts as the structural hub of the relational schema. Almost every business event revolves around an order, and this table provides the order_id primary key that links outward to the customer who made the purchase, the items bought, the payments made, and the reviews left.

3. What is the difference between customer_id and customer_unique_id?

    customer_id is a transactional identifier. A new customer_id is generated every single time an order is placed, acting as a unique key for that specific purchase instance.

    customer_unique_id is the actual identifier for the individual person. If the same person makes three separate purchases over a year, they will have one customer_unique_id but three different customer_ids attached to their orders.

4. Which columns appear to represent dates?
Several columns across different tables represent datetime objects:

    Orders table: order_purchase_timestamp, order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date.

    Order Items table: shipping_limit_date.

    Reviews table: review_creation_date, review_answer_timestamp.

5. Which columns appear to represent categorical variables?
Categorical columns represent discrete groupings or states. Examples include:

    order_status (e.g., delivered, shipped, canceled, processing)

    payment_type (e.g., credit_card, boleto, voucher, debit_card)

    product_category_name (e.g., health_beauty, sports_leisure)

    customer_state and seller_state (e.g., SP, RJ, MG)

6. Which columns contain missing values?
In the Olist dataset, missing values typically appear in:

    Timestamp columns: order_approved_at, order_delivered_carrier_date, and order_delivered_customer_date are naturally missing if an order was canceled or hasn't shipped yet.

    Review text: review_comment_title and review_comment_message have many nulls because leaving a written review is optional.

    Product attributes: product_category_name, product_weight_g, and physical dimensions are missing for a small subset of unregistered products.

7. Which tables can be joined using order_id?
The order_id key allows you to join the central olist_orders_dataset with:

    olist_order_items_dataset (to see what was bought)

    olist_order_payments_dataset (to see how it was paid for)

    olist_order_reviews_dataset (to see the customer satisfaction)

8. Which tables can be joined using product_id?
The product_id key connects the transactional item data to the static catalog data. It joins:

    olist_order_items_dataset (the transaction level)

    olist_products_dataset (the catalog level, containing dimensions and categories)

9. What are three potential Machine Learning problems that can be created from this dataset?

    Delivery Time Prediction (Regression): Predicting the actual delivery time using the product's dimensions, the seller's location, and the customer's location.

    Customer Satisfaction Prediction (Classification): Predicting whether an order will receive a 1-star or 5-star review based on freight value, delivery delays, and product category.

    Customer Lifetime Value / Churn (Predictive Analytics): Identifying which customer_unique_ids are highly likely to make a repeat purchase based on their initial buying behavior and payment methods.

10. What ethical concerns may arise when using customer data for ML?

    Privacy and De-anonymization: Even if names are removed, combining unique identifiers with precise locations (like zip codes and time of purchase) can sometimes be reverse-engineered to identify real individuals.

    Algorithmic Bias: If an ML model is trained to optimize delivery routes or marketing spend, it might learn to systematically ignore rural or lower-income states (which may historically have longer delivery times or lower sales), unfairly penalizing users in those regions.

    Informed Consent: Customers agreed to provide their data to receive a product, but they may not have explicitly consented to have their purchasing habits modeled for behavioral profiling.
