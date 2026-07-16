# Data Dictionary

## 1. customers

| Column | Data Type | Description | Example Value |
| --- | --- | --- | --- |
| customer_id | String | Unique customer identifier for each order | 06b8999e2fba1a1fbc88172c00ba8bc7 |
| customer_unique_id | String | Unique identifier for the customer | 861eff4711a542e4b93843c6dd7febb0 |
| customer_zip_code_prefix | Integer | Zip code of customer | 14409 |
| customer_city | String | City of customer | franca |
| customer_state | String | State of customer | SP |

## 2. orders

| Column | Data Type | Description | Example Value |
| --- | --- | --- | --- |
| order_id | String | Unique order identifier | e481f51cbdc54678b7cc49136f2d6af7 |
| customer_id | String | Key to the customer dataset | 9ef432eb6251297304e76186b10a928d |
| order_status | String | Current status of order | delivered |
| order_purchase_timestamp | Timestamp | Shows the purchase timestamp | 2017-10-02 10:56:33 |
| order_approved_at | Timestamp | Shows the payment approval timestamp | 2017-10-02 11:07:15 |

## 3. order_items

| Column | Data Type | Description | Example Value |
| --- | --- | --- | --- |
| order_id | String | Unique order identifier | 00010242fe8c5a6d1ba2dd792cb16214 |
| order_item_id | Integer | Sequential number identifying number of items included in the same order | 1 |
| product_id | String | Product identifier | 4244733e06e7ecb4970a6e2683c13e61 |
| seller_id | String | Seller identifier | 48436dade18ac8b2bce089ec2a041202 |
| price | Float | Item price | 58.90 |
