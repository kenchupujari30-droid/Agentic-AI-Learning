from dataclasses import dataclass
from pydantic import BaseModel

# Option 1: dataclass — clean syntax, ZERO validation
@dataclass
class UserDataClass:
  name: str
  email:str
  age: int
user =  UserDataClass(name = "Kenchu", email = "Kenchu@gmail.com", age = "not a number" )
print(user.age)  

# Option 2: BaseModel — actually inspects the data
class UserModel(BaseModel):
    name: str
    email: str
    age: int

UserModel(name="Advik", email="Advik@gmail.com", age="not a number")
# ValidationError: Input should be a valid integer,
# unable to parse string as an integer

# But a numeric STRING is coerced safely:
user = UserModel(name="Alice", email="alice@example.com", age="30")
print(user.age, type(user.age))   # 30 <class 'int'>