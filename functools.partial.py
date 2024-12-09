from functools import partial
def greet(greeting,name):
    print(f'{greeting}, {name}!')

greet_hello = partial(greet, greeting='hello')


greet_morning = partial(greet_hello, name="Charlie")
greet_morning()

greet_hello(name="Bob")
greet_hello(name="Alice")

