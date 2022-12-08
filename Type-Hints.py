"""Type hintsの練習"""

from typing import List, Dict, Union, TypedDict, Any
from dataclasses import dataclass

age: int = 19
answer: bool = True
name: str = 'aaaa'
pi: float = 3.14

test_record_first: list = [50, 80, 49, 94, 42]
number: dict = {1 :"鴉さん", 2: '阿井さん'}

test_record_second: List[int] = [99, 97, 89, 80, 79]
number: Dict[Dict] = {1 :"鴉さん", 2: '阿井さん'}

practice: Union[int, str] = 10

def multiple(x: int, y: int) -> int:
    return x * y

def number_print(number: int) -> None:
    print(f'{number}です')
    
Movie = TypedDict('Movie' , {'name': str, 'year': int})

movie: Movie = {'name': 'your name', 'year': 2000}

@dataclass
class Student:
    name: str
    student_number: str

    def print_information(self) -> None:
        print(f'Name: {self.name}, Student Number: {self.student_number}')

test: Any = 'Hello'