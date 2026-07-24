from datetime import datetime

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculator(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return "Error: Invalid mathematical expression"