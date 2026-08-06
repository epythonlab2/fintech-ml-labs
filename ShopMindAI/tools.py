# tools.py
from langchain_core.tools import tool
from mock_db import ORDERS_DB

@tool
def get_order_status(order_id: str) -> str:
    """Fetch real-time shipment status and item info for a given order ID."""
    order = ORDERS_DB.get(order_id.upper())
    if not order:
        return f"Order {order_id} not found in ShopMind AI database."
    return f"Order {order_id}: {order['item']} - Status: {order['status']}"

@tool
def process_return(order_id: str) -> str:
    """Evaluates if an order is within the 30-day return window and calculates refund."""
    order = ORDERS_DB.get(order_id.upper())
    if not order:
        return f"Order {order_id} not found."
    
    if order["days_old"] > 30:
        return f"Return rejected for {order_id}. Order is {order['days_old']} days old (limit: 30 days)."
    
    return f"Return approved for {order_id}. Refund of ${order['price']} initiated."