from __future__ import annotations

'''a module to implement basic ideas involving complex numbers'''

from functools import reduce
from itertools import combinations
from typing import List
import math


class Complex (object):
    '''
       represents a complex number using the idea that a complex number
       is simply an ordered pair of real numbers with the usual component-wise
       operation of addition, and a specialized multiplication operation whose 
       geometric significance only becomes clear when polar coordinates are used.

       Example usage:
    
       c = Complex (1, 2) # represents the complex number 1 + 2i
    '''
    def __init__ (self, x: float, y: float):
        self.x = x
        self.y = y
     
    def __add__ (self, other: Complex) -> Complex:
        '''
           returns the sum of two complex numbers.

           Example usage:

           c1 = Complex (1, 2)
           c2 = Complex (3, 4)

           c1 + c2 == Complex (4, 6)
        '''
        return Complex (other.x + self.x, other.y + self.y)

    def __sub__ (self, other) -> Complex:
        '''
           returns the difference of two complex numbers.

           Example usage:

           c1 = Complex (1, 2)
           c2 = Complex (3, 4)

           c1 - c2 == Complex (-2, -2)
        '''
        return Complex (self.x - other.x, self.y - other.y)

    def __mul__ (self, other: Complex) -> Complex:
        '''
           returns the product of two complex numbers.

           Example usage:

           c1 = Complex (0, 1)
           c2 = Complex (0, 1)

           c1 * c2 == Complex (-1, 0)
        '''
        return Complex (self.x * other.x - self.y * other.y, self.x * other.y + self.y * other.x)
     
    def __truediv__ (self, other: Complex) -> Complex:
        '''
           returns the quotient of two complex numbers using the complex conjugate of the denominator.

           Example usage:

           c1 = Complex (1, 2)
           c2 = Complex (3, 4)

           c1 / c2 == Complex (0.44, 0.08)
        '''
        numerator = self * other.complex_conjugate ()
        denominator = other * other.complex_conjugate ()
        return numerator.scalar_multiplication (1/denominator.x)

    def __neg__ (self) -> Complex:
        '''
           returns the negation of this complex number.

           Example usage:

           c1 = (1, 2) 
           -c1 == Complex (-1, -2)
        '''
        return Complex (-self.x, -self.y)

    def __eq__ (self, other, epsilon=0.1) -> Complex:
        '''
           returns True if two complex numbers lie within `epsilon` of each other, False otherwise.
        
           Example usage:

           c1 = Complex (2.001, -5.001)
           c2 = Complex (2, -5)
           c1 == c2 # True
        '''
        return abs (self.x - other.x) < epsilon and abs (self.y - other.y) < epsilon

    def sum_all (self, list_of_complex_numbers) -> Complex:
        '''
           returns the result of summing this complex number with a list of complex numbers.

           Example usage:

           c1 = Complex (1, 2)
           c2 = Complex (3, 4)
           c3 = Complex (5, 6)
           c1.sum_all ([c2, c3]) == Complex (8, 10)
        '''
        return reduce (lambda x, y: x + y, list_of_complex_numbers)

    def scalar_multiplication (self, scalar) -> Complex:
        '''
           returns the result of scalar multiplciaiotn of this Copmlex object with a scalar value.

           Example usage:

           c1 = Complex (1, 2)   
           c1.scalar_multiplication (2) == Complex (2, 4)
        '''
        return Complex (scalar * self.x, scalar * self.y)
    
    def complex_conjugate (self) -> Complex:
        '''
           returns the result of complex conjugating a complex number.

           Example usage:

           c1 = Complex (1, 2)
           c1.complex_conjugate () == Complex (1, -2)    
        '''
        return Complex (self.x , -self.y)
    
    def modulus (self) -> float:
        '''
           returns the magnitude of this complex number.

           Example usage:

           c1 = Complex (0, 1)
           c1.modulus () == 1.0
        '''
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def modulus_squared (self) -> float:
        '''
           returns the squared magnitude of this complex number as a single, real number.

           Example usage:

           c1 = Complex (1, 2)
           c1.modulus_squared () == 5
        '''
        return (self * self.complex_conjugate()).x
    
    def __str__ (self) -> str:
        return f'{self.x} + {self.y}i'
    
    def __repr__ (self) -> str:
        return f'({self.x}, {self.y})'

class ComplexVector (object):
    '''
       represents a vector of complex numbers.

       Example usage:

       c1 = Complex (1, 2)
       c2 = Complex (3, 4)

       v = ComplexVector ([c1, c2])       
    '''
    def __init__ (self, list_of_complex_numbers: List[Complex]):
        self.list_of_complex_numbers = list_of_complex_numbers
        self.current_index = 0
    
    def hermitian_conjugate (self) -> ComplexVector:
        '''
           returns the hermitian conjugate of a ComplexVector.

           Example usage:

           c1 = Complex (1, 2)
           c2 = Complex (3, 4)

           v = ComplexVector ([c1, c2])
           v.hermitian_conjugate () == ComplexVector ([(1, -2),(3, -4)])
        '''
        return ComplexVector ([v.complex_conjugate () for v in self])

    def inner_product (self, other: ComplexVector) -> Complex:
        '''
           returns the inner product of this complex vector with some other complex vector.

           Example usage:

           c1 = Complex (1, 2)
           c2 = Complex (3, 4)

           v = ComplexVector ([c1, c2])
           v.inner_product (v) == Complex (30, 0)
        '''
        sum_so_far = Complex (0,0)
        bra = self.hermitian_conjugate ()
        product_pairs = zip (bra, other)
        for z1, z2 in product_pairs:
            sum_so_far += z1 * z2
        return sum_so_far

    def norm (self) -> float:
        '''
           returns the norm of this complex vector.

           Example usage:

           c1 = Complex (1, 2)
           c2 = Complex (3, 4)

           v = ComplexVector ([c1, c2])
           v.norm () == 5.477225575051661
        '''
        return math.sqrt (self.inner_product(self).x)

    def normalize (self) -> ComplexVector:
        '''
           returns a normalized version of this vector.

           Example usage:

           c1 = Complex (1, 2)
           c2 = Complex (3, 4)

           v = ComplexVector ([c1, c2])
           v.normalize () == ComplexVector ([Complex (0.18257418583505536, 0.3651483716701107), Complex (0.5477225575051661, 0.7302967433402214)])
        '''
        norm = self.norm ()
        return self.scalar_multiplication (1 / norm)

    def is_normalized (self, epsilon=0.1) -> bool:
        '''
           returns True if this vector is normalized, False otherwise.

           Example usage:

           v = ComplexVector ([Complex (0.18257418583505536, 0.3651483716701107), Complex (0.5477225575051661, 0.7302967433402214)])
           v.is_normalized () == True
        '''
        return abs (self.norm() - self.normalize().norm()) < epsilon

    def is_orthogonal_to (self, other, epsilon=0.1) -> bool:
        '''
           returns True if this vector is orthogonal to the other vector, False otherwise.

           Example usage:

           c1 = Complex (2, 3)
           c2 = Complex (4, -2)
           c3 = Complex (1, 1)
           c4 = Complex (1, 1)
           c5 = Complex (1, -1)
           c6 = Complex (2, 3)
           c7 = Complex (4, -6)
           c8 = Complex (1, 0)

           v1 = ComplexVector ([c1, c2, c3, c4])
           v2 = ComplexVector ([c5, c6, c7. c8])

           v1.is_orthogonal_to (v2) == True
        '''
        return abs (self.inner_product(other).modulus()) < epsilon

    def projection_onto (self, other: ComplexVector) -> ComplexVector:
        '''
           returns the projection of this complex vector onto some other complex vector.
        '''
        unit_vector = other.normalize()
        return unit_vector.scalar_multiplication(self.inner_product(unit_vector))

    def __add__ (self, other: ComplexVector) -> ComplexVector:
        '''returns the sum of two complex vectors'''
        pairs = zip (self, other)
        return ComplexVector ([u + v for (u, v) in pairs])

    def sum_all (self, list_of_complex_vectors):
        '''returns the result of summing this ComplexVector with a list of ComplexVectors'''
        return reduce ((lambda x, y: x + y), list_of_complex_vectors)

    def __sub__ (self, other) -> ComplexVector:
        '''returns the result of subtracting this complex vector from antoher complex vector'''
        pairs = zip (self, other)
        return ComplexVector ([u - v for (u, v) in pairs])

    def __mul__(self, other) -> ComplexVector:
        '''returns the component-wise product of two complex vectors'''
        pairs = zip (self, other)
        return ComplexVector ([u * v for (u, v) in pairs])

    def __truediv__ (self, other: ComplexVector) -> ComplexVector:
        '''returns the component-wise division of two complex vectors'''
        pairs = zip (self, other)
        return ComplexVector ([u / v for (u, v) in pairs])

    def scalar_multiplication (self, scalar: float):
        '''returns the component-wise scalar multiplication of a complex vector with a scalar'''
        if type(scalar) == Complex:
            return ComplexVector ([u * scalar for u in self.list_of_complex_numbers]) 
        return ComplexVector ([u.scalar_multiplication (scalar) for u in self.list_of_complex_numbers])
    
    def complex_conjugate (self):
        '''returns the component-wise complex-conjugate of a complex vector'''
        return ComplexVector ([u.complex_conjugate () for u in self.list_of_complex_numbers])

    def __len__ (self):
        return len (self.list_of_complex_numbers)

    def __iter__ (self):
        '''allows complex vectors to be used with the zip() function'''
        return (x for x in self.list_of_complex_numbers)

    def __next__ (self):
        '''allows complex vectors to be used in for loops'''
        try: 
            current = self.list_of_complex_numbers[self.current_index]
        except IndexError:
            raise StopIteration
        self.current_index += 1
        return current
        
    def __str__ (self):
        list_of_strings = [str(c) for c in self.list_of_complex_numbers]
        return '[' + ','.join(list_of_strings) + ']'
     
    def __repr__ (self):
        list_of_strings = [repr(c) for c in self.list_of_complex_numbers]
        return '[' + ', '.join(list_of_strings) + ']'

def sum_all (list_of_complex_vectors: List[ComplexVector]) -> ComplexVector:
    '''returns the sum of a list of ComplexVectors'''
    return reduce ((lambda x, y: x + y), list_of_complex_vectors)

def gram_schmidt (list_of_vectors: List[ComplexVector]):
    '''given a a linearly independent set of complex vectors, returns an orthonormal basis'''
    if len (list_of_vectors) == 0:
        return []

    if len (list_of_vectors) == 1:
        return [v.normalize () for v in list_of_vectors]

    # grab the last vector from the list of vectors
    # u0 = list_of_vectors[0]

    # for v, index in enumerate (list_of_vectors[1:]):

    u = list_of_vectors[0]

    # orthonormalize the rest of the vectors
    basis = gram_schmidt (list_of_vectors[1:])

    # compute the last orthonormal basis vector
    w = (u - sum_all(ComplexVector(u.projection_onto(v)  for v in basis)))

    # append this orthonormal vector to the rest of the orthonormal basis
    basis.append (w.normalize())
    return basis

def are_normalized (list_of_vectors):
    '''given a list of ComplexVectors, returns True if that set is normal, False otherwise'''
    return all([v.is_normalized() for v in list_of_vectors])

def are_orthogonal (list_of_vectors):
    '''given a list of ComplexVectors, returns True if that set is orthogonal, False otherwise'''
    pairs = [pair for pair in combinations(list_of_vectors, 2)]
    return all([v1.is_orthogonal_to(v2) for v1, v2 in pairs])

def are_orthonormal (list_of_vectors):
    '''given a list of ComplexVectors, returns True if that set is orthonormal, False otherwise:
       
       v1 = ComplexVector([Complex(1/math.sqrt(2), 0), Complex(1/math.sqrt(2), 0)])
       v2 = ComplexVector([Complex(1/math.sqrt(2), 0), Complex(-1/math.sqrt(2), 0)])
       are_orthonormal([v1, v2]) # true
    '''
    return are_normalized (list_of_vectors) and are_orthogonal(list_of_vectors)

def linear_combination (list_of_vectors, list_of_scalars):
    '''given several lists of Complex objects, or several ComplexVectors, and a list of scalars,
       returns the corresponding linear combination as a ComplexVector:
    
      v1 = [Complex(-1,0), Complex(0,7), Complex(2,0)]
      v2 = [Complex(0,0), Complex(2,0), Complex(4,0)]
      list_of_vectors = [v1, v2]
      list_of_scalars = [1, 2]
      linear_combination(list_of_vectors, list_of_scalars)

      or:

      v1 = ComplexVector([Complex(-1,0), Complex(0,7), Complex(2,0)])
      v2 = ComplexVector([Complex(0,0), Complex(2,0), Complex(4,0)])
      list_of_vectors = [v1, v2]
      linear_combination(list_of_vectors, [1,2])
    '''
    length = len (list_of_vectors[0])
    vectors_and_scalars = list(zip(list_of_vectors, list_of_scalars))
    scaled_vectors = []
    for vector_scalar in vectors_and_scalars:
        scalar = vector_scalar[-1]
        vector = vector_scalar[:-1][0]
        new_vector = [u.scalar_multiplication(scalar) for u in vector]
        scaled_vectors.append(new_vector)
    vectors_to_add = [(x[0], x[1]) for x in list(zip(*scaled_vectors))]
    answer= []
    for i in range(length):
        answer.append(reduce(Complex.__add__, vectors_to_add[i]))
    return ComplexVector(answer)
