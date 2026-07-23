# The four basic types you'll use constantly
name: str = "Kenchu"
age: int = 28
price: float = 499.99
is_active: bool = True

# Container types
tags: list[str] = ["python", "pydantic", "fastapi"]
word_counts: dict[str, int] = {"error": 12, "warning": 5}

# Function signatures — the most valuable place to use hints
def format_price(amount: float, currency: str = "USD") -> str:
    return f"{currency} {amount:.2f}"

# The gotcha: Python enforces NONE of this
# age =   "not a number at all"   # runs. no error. no warning.
age = 30
print(f"Intager na a number:  {age}")