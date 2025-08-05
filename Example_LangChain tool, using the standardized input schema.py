from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# 1.  Define the standardized Pydantic model for the tool's input
class StandardItem(BaseModel):
    product_name: str = Field(description="The name of the product")
    quantity: int = Field(description="The quantity of the product")
    price_per_unit: Optional[float] = Field(default=None, description="The price of each unit (optional)")

class OrderInput(BaseModel):
    items: List[StandardItem] = Field(description="A list of items in the order, with standardized column names.")

# 2. Define a function to preprocess and standardize the input
def preprocess_items(input_items: List[Dict]) -> List[StandardItem]:
    """
    Preprocesses a list of dictionaries, handling varying column names and
    converting them to the standard 'StandardItem' format.
    """
    standardized_items = []
    # Define a mapping for potential column name variations
    column_name_mapping = {
        "item_name": "product_name",
        "product": "product_name",
        "name": "product_name",
        "qty": "quantity",
        "amount": "quantity",
        "unit_price": "price_per_unit",
        "price": "price_per_unit",
    }

    for item in input_items:
        processed_item = {}
        for key, value in item.items():
            # Standardize column names
            standard_key = column_name_mapping.get(key.lower(), key.lower())  # Use .lower() for case-insensitivity
            processed_item[standard_key] = value

        # Create StandardItem instance, allowing for missing optional fields
        try:
            standardized_items.append(StandardItem(**processed_item))
        except Exception as e:
            # Handle cases where mandatory fields might be missing after pre-processing
            print(f"Error processing item: {item}. Reason: {e}") # Debugging aid
            # You might choose to skip the item, raise an error, or try to infer missing values
            pass

    return standardized_items

# 3.  Define the LangChain tool, using the standardized input schema
@tool(args_schema=OrderInput)
def process_dynamic_order(items: List[StandardItem]) -> str:
    """
    Processes a list of items with standardized column names (product_name, quantity, price_per_unit)
    and returns a summary message.
    """
    total_value = 0
    item_details = []
    for item in items:
        item_details.append(f"{item.quantity} x {item.product_name}")
        if item.price_per_unit:
            total_value += item.quantity * item.price_per_unit
    
    summary = f"Order processed for: {', '.join(item_details)}."
    if total_value > 0:
        summary += f" Total estimated value: ${total_value:.2f}."
    return summary

# 4. Example Usage: Simulate different input formats
if __name__ == "__main__":
    # Example 1: Standard input format
    standard_input = [
        {"product_name": "Apple", "quantity": 5, "price_per_unit": 0.50},
        {"product_name": "Banana", "quantity": 2}, # Missing optional field
    ]
    preprocessed_standard = preprocess_items(standard_input)
    print(f"Standard Input Result: {process_dynamic_order(preprocessed_standard)}")

    # Example 2: Input with varying column names
    dynamic_input = [
        {"item_name": "Orange", "qty": 3, "price": 0.75},
        {"product": "Grapes", "amount": 1, "price_per_unit": 2.20},
        {"name": "Mango", "quantity": 4},
    ]
    preprocessed_dynamic = preprocess_items(dynamic_input)
    print(f"Dynamic Input Result: {process_dynamic_order(preprocessed_dynamic)}")

    # Example 3: Input with completely unknown column names (handled by 'extra = Extra.allow')
    # This example assumes you want to allow unknown fields and only process the known ones
    # For a stricter approach, you might remove unknown fields during preprocessing
    unknown_input = [
        {"product_name": "Milk", "quantity": 1, "extra_info": "dairy aisle"},
        {"juice": "Apple", "size": "small"}, # This item will likely fail validation if 'product_name' and 'quantity' are required
    ]
    preprocessed_unknown = preprocess_items(unknown_input)
    print(f"Unknown Input Result: {process_dynamic_order(preprocessed_unknown)}")

    # Example 4: Input with Pydantic's 'Extra.allow' (directly if you want to allow extra fields)
    class ItemWithExtra(BaseModel):
        product_name: str
        quantity: int
        class Config:
            extra = 'allow' # Allows additional fields not defined in the model

    @tool(args_schema=ItemWithExtra)
    def process_with_extra(product_name: str, quantity: int, **kwargs) -> str:
        """Processes an item, allowing for extra fields."""
        return f"Processed {quantity} of {product_name} (with extra info: {kwargs})"

    # Note: When using Extra.allow, the LLM will still be guided by the defined schema,
    # but if it generates extra fields, they will be passed through to your function's kwargs.
    # The agent might still prefer to stick to the defined schema unless explicitly prompted to do otherwise.

max(float(item.get("Potential Cost Savings", 0)) for item in opportunities)

For example, replace imports like: `from langchain_core.pydantic_v1 import BaseModel`
with: `from pydantic import BaseModel`



        opportunities = eval(opportunities)
        max_cost = max(float(item.get("Potential Cost Savings", 0)) for item in opportunities)
        max_drivers = max(len(item.get("Potential Drivers", [])) for item in opportunities)
        
        #eval(opportunities)
results = []
for item in opportunities:
            signals = extract_signals(item, max_cost, max_drivers)
            score = sum(signals[key] * weights[key] for key in weights)
            results.append({
                "Opportunity Title": item["Opportunity Title"],
                "Smart Opportunity Score": round(score, 4)
            })

        return results
