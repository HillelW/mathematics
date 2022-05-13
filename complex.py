import math
from functools import reduce
from itertools import combinations


'''a module to implement basic ideas in quantum computing'''

def sum_all(list_of_complex_vectors):
    '''returns the sum of a list of ComplexVectors'''
    return reduce((lambda x, y: x + y), list_of_complex_vectors)

def gram_schmidt(list_of_vectors):
    '''given a basis of ComplexVectors, returns an orthonormal basis'''
    if len(list_of_vectors) == 0:
        return []

    if len(list_of_vectors) == 1:
        return [v.normalize() for v in list_of_vectors]

    # grab the last vector from the list of vectors
    u = list_of_vectors[-1]

    # orthonormalize the rest of the vectors
    basis = gram_schmidt(list_of_vectors[0: -1])

    # compute the last orthonormal basis vector
    w = (u - sum_all(ComplexVector(u.projection_onto(v)  for v in basis)))

    # append this orthonormal vector to the rest of the orthonormal basis
    basis.append(w.normalize())

    return basis

def are_normalized(list_of_vectors):
    '''given a list of ComplexVectors, returns True if that set is normal, False otherwise'''
    return all([v.is_normalized() for v in list_of_vectors])

def are_orthogonal(list_of_vectors):
    '''given a list of ComplexVectors, returns True if that set is orthogonal, False otherwise'''
    pairs = [pair for pair in combinations(list_of_vectors, 2)]
    return all([v1.is_orthogonal_to(v2) for v1, v2 in pairs])

def are_orthonormal(list_of_vectors):
    '''given a list of ComplexVectors, returns True if that set is orthonormal, False otherwise:
       
       v1 = ComplexVector([Complex(1/math.sqrt(2), 0), Complex(1/math.sqrt(2), 0)])
       v2 = ComplexVector([Complex(1/math.sqrt(2), 0), Complex(-1/math.sqrt(2), 0)])
       are_orthonormal([v1, v2]) # true
    '''
    return are_normalized(list_of_vectors) and are_orthogonal(list_of_vectors)

def linear_combination(list_of_vectors, list_of_scalars):
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
    length = len(list_of_vectors[0])
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

class Complex():
    '''represents a complex number:
    
       c = Complex(1, 2)
    '''
    def __init__(self, x, y=0.0):
        self.x = x
        self.y = y
     
    def __add__(self, other):
        '''returns the sum of two complex numbers'''
        return Complex(other.x + self.x, other.y + self.y)

    def sum_all(self, list_of_complex_numbers):
        '''returns the result of summing this complex number with a list of complex numbers'''
        return reduce((lambda x, y: x + y), list_of_complex_numbers)

    def __sub__(self, other):
        '''returns the result of subtracting this complex number from antoher complex number'''
        return Complex(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        '''returns the product of two complex numbers'''
        return Complex(self.x * other.x - self.y * other.y, self.x * other.y + self.y * other.x)
     
    def __truediv__(self, other):
        '''returns the quotient of two complex numbers using the complex conjugate of the denominator'''
        numerator = self * other.complex_conjugate()
        denominator = other * other.complex_conjugate()
        return numerator.scalar_multiplication(1/denominator.x)

    def scalar_multiplication(self, scalar):
        '''returns the result of scalar multiplciaiotn of this Copmlex object with a scalar value'''
        return Complex(scalar * self.x, scalar * self.y)
    
    def complex_conjugate(self):
        '''returns the result of complex conjugation'''
        return Complex(self.x , -self.y)
    
    def modulus(self):
        '''returns the magnitude of this complex number'''
        return math.sqrt(self.x ** 2 + self.y ** 2)
    
    def modulus_squared(self):
        '''returns the squared magnitude of this complex number as a single, real number'''
        return (self * self.complex_conjugate()).x

    def __neg__(self):
        '''returns the negation of this complex number'''
        return Complex(-self.x, -self.y)

    def __eq__(self, other, epsilon=0.1):
        '''returns True if each component falls within the epsilon band, False otherwise.
        
        c1 = Complex(2.001, -5.001)
        c2 = Complex(2, -5)
        c1 == c2 # True
        '''
        return abs(self.x - other.x) < epsilon and abs(self.y - other.y) < epsilon
    
    def __str__(self):
        return f'{self.x} + {self.y}i'
    
    def __repr__(self):
        return f'({self.x}, {self.y})'

class ComplexVector():
    '''represents a list of Copmlex objects'''
    def __init__(self, list_of_complex_numbers):
        self.list_of_complex_numbers = list_of_complex_numbers
        self.current_index = 0
    
    def hermitian_conjugate(self):
        '''returns the hermitian conjugate of a ComplexVector'''
        return ComplexVector([v.complex_conjugate() for v in self])

    def inner_product(self, other):
        '''returns the inner product of this vector with some other vector'''
        sum_so_far = Complex(0,0)
        bra = self.hermitian_conjugate()
        product_pairs = zip(bra, other)
        for z1, z2 in product_pairs:
            sum_so_far += z1 * z2
        return sum_so_far

    def norm(self):
        '''returns the norm of this vector'''
        return math.sqrt(self.inner_product(self).x)

    def normalize(self):
        '''returns a normalized version of this vector'''
        norm = self.norm()
        return self.scalar_multiplication(1/norm)

    def is_normalized(self, epsilon=0.1):
        '''returns True if this vector is normalized, False otherwise'''
        return abs(self.norm() - self.normalize().norm()) < epsilon

    def is_orthogonal_to(self, other, epsilon=0.1):
        '''returns True if this vector is orthogonal to the other vector, False otherwise'''
        return abs(self.inner_product(other).modulus()) < epsilon

    def projection_onto(self, other):
        '''returns the projection of this vector onto some other vector'''
        unit_vector = other.normalize()
        return unit_vector.scalar_multiplication(self.inner_product(unit_vector))

    def __add__(self, other):
        '''returns the sum of two complex vectors'''
        pairs = zip(self, other)
        return ComplexVector([u + v for (u, v) in pairs])

    def sum_all(self, list_of_complex_vectors):
        '''returns the result of summing this ComplexVector with a list of ComplexVectors'''
        return reduce((lambda x, y: x + y), list_of_complex_vectors)

    def __sub__(self, other):
        '''returns the result of subtracting this complex vector from antoher complex vector'''
        pairs = zip(self, other)
        return ComplexVector([u - v for (u, v) in pairs])

    def __mul__(self, other):
        '''returns the component-wise product of two complex vectors'''
        pairs = zip(self, other)
        return ComplexVector([u * v for (u, v) in pairs])

    def __truediv__(self, other):
        '''returns the component-wise division of two complex vectors'''
        pairs = zip(self, other)
        return ComplexVector([u / v for (u, v) in pairs])

    def scalar_multiplication(self, scalar):
        '''returns the component-wise scalar multiplication of a complex vector with a scalar'''
        if type(scalar) == Complex:
            return ComplexVector([u * scalar for u in self.list_of_complex_numbers]) 
        return ComplexVector([u.scalar_multiplication(scalar) for u in self.list_of_complex_numbers])
    
    def complex_conjugate(self):
        '''returns the component-wise complex-conjugate of a complex vector'''
        return ComplexVector([u.complex_conjugate() for u in self.list_of_complex_numbers])

    def __len__(self):
        return len(self.list_of_complex_numbers)

    def __iter__(self):
        '''allows complex vectors to be used with the zip() function'''
        return (x for x in self.list_of_complex_numbers)

    def __next__(self):
        '''allows complex vectors to be used in for loops'''
        try: 
            current = self.list_of_complex_numbers[self.current_index]
        except IndexError:
            raise StopIteration
        self.current_index += 1
        return current
        
    def __str__(self):
        list_of_strings = [str(c) for c in self.list_of_complex_numbers]
        return '[' + ','.join(list_of_strings) + ']'
     
    def __repr__(self):
        list_of_strings = [repr(c) for c in self.list_of_complex_numbers]
        return '[' + ','.join(list_of_strings) + ']'
