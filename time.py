
from datetime import datetime

timestamp = 1545730073
dt_object = datetime.fromtimestamp(timestamp)

print("dt_object =", dt_object)
print("type(dt_object) =", type(dt_object))

print(dt_object.year)
print(dt_object.month)
print(dt_object.day)
print(dt_object.hour)
print(dt_object.weekday())

