def register_user(name, email, age):
  birth_year = 2026 - age  # assumes age is an int — nothing enforces that
  print(f"Registered {name}, born approx. {birth_year}")

register_user("Advik", "advik@example.com", 15)  
register_user("Kenchu", "Kenchu@example.com", "unknown")  


#  python src/register_user.py
# Registered Advik, born approx. 2011
# Traceback (most recent call last):
#   File "E:\Agentic_Ai_Learning\04_Pydantic\src\register_user.py", line 6, in <module>
#     register_user("Kenchu", "Kenchu@example.com", "unknown")  
#     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "E:\Agentic_Ai_Learning\04_Pydantic\src\register_user.py", line 2, in register_user
#     birth_year = 2026 - age  # assumes age is an int — nothing enforces that
#                  ~~~~~^~~~~
# TypeError: unsupported operand type(s) for -: 'int' and 'str'
# (04_Pydantic) 