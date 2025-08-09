import numpy as np
import pandas as pd 


# ---

# ## 🧩 1. Array Creation
# **Question:** Create the following arrays using NumPy:
# 1. A 1D array with numbers from 0 to 9
# 2. A 2D array of shape (3, 3) filled with zeros
# 3. A 2D array of shape (2, 4) filled with ones
# 4. A 3x3 identity matrix
arr1=np.arange(10)
print(arr1)

arr2=np.zeros((3,3))
print(arr2)

arr3=np.ones((2,4))
print(arr3)

arr4=np.eye(3)
print(arr4)

# ---






# ## 🧪 2. Array Inspection
# **Question:** Given a NumPy array `arr`, print:
# 1. Its shape
# 2. Its data type
# 3. The number of dimensions
# 4. The total number of elements
print("#"* 50)
# ```python
# arr = np.array([[1, 2, 3], [4, 5, 6]])
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(np.shape(arr))
print(arr.dtype)
print(arr.ndim)
print(arr.size)

# ## 🔍 3. Indexing and Slicing
# **Question:** Using the array below, do the following:
# 1. Slice the first two rows
# 2. Get the last column
# 3. Extract the element in the second row and third column
# 4. Reverse the rows

# ```python
# arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])

arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print(arr[:2])
print(arr[:, -1])
print(arr[1, 2])
print(arr[::-1])



# ---

# ## 🔁 4. More Slicing Practice
# **Question:**
# Given the array below, extract:
# 1. All even numbers
# 2. The second and third rows
# 3. The last two columns
# 4. A subarray containing the middle 2x2 block

arrr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]])

# 1. All even numbers
print(arrr[arrr % 2 == 0])

# 2. The second and third rows
print(arrr[1:3])

# 3. The last two columns
print(arrr[:, -2:])

# 4. A subarray containing the middle 2x2 block
print(arrr[1:3, 1:3])

# ---

# ## ➕ 5. Arithmetic Operations
# **Question:** Perform the following using NumPy arrays:
# 1. Add two arrays element-wise
# 2. Multiply two arrays element-wise
# 3. Raise all elements of an array to the power of 2

# ```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# 1. Add two arrays element-wise
print(arr1 + arr2)

# 2. Multiply two arrays element-wise
print(arr1 * arr2)

# 3. Raise all elements of an array to the power of 2
print(arr1 ** 2)

# ## 📏 6. Broadcasting
# **Question:** Use broadcasting to:
# 1. Add a 1D array to each row of a 2D array
# 2. Multiply each row of a 2D array by a 1D array
# 3. Subtract a scalar from each element in an array
# 4. Add a column vector to each column of a matrix

A = np.array([[1, 2, 3],
              [4, 5, 6]])

B = np.array([10, 20, 30])
C = np.array([[1], [2]])
# 1. Add a 1D array to each row of a 2D array
print(A + B)
# 2. Multiply each row of a 2D array by a 1D array
print(A * B)
# 3. Subtract a scalar from each element in an array
print(A - 5)
# 4. Add a column vector to each column of a matrix
print(A + C)




# ---

# ## 🔁 7. More Broadcasting Practice
# **Question:** Try these:
# 1. Add a 1D array of shape (4,) to a 2D array of shape (3,4)
# 2. Multiply a (3,1) column array to a (1,4) row array and observe the shape
# 3. Add a scalar to the entire array


A = np.arange(12).reshape(3, 4)
B = np.array([100, 200, 300, 400])
C = np.array([[2], [3], [4]])
# 1. Add a 1D array of shape (4,) to a 2D array of shape (3,4)
print(A + B)
# 2. Multiply a (3,1) column array to a (1,4) row array and observe the shape
print(C * B.reshape(1, 4))
# 3. Add a scalar to the entire array
print(A + 10)






# ## 🔃 8. Reshaping and Flattening
# **Question:**
# 1. Reshape a 1D array of 12 elements into a 3x4 matrix
# 2. Flatten a 2D array into a 1D array


arr6 = np.arange(12).reshape(3, 4)
print(arr6)
flattened_arr = arr6.flatten()
print(flattened_arr)






# ---

# ## 🔗 9. Stacking and Splitting
# **Question:** Stack and split arrays:
# 1. Stack two (2, 2) arrays vertically and horizontally
# 2. reshape a (4, 4) array into array of shape (2, 8)

print("#"* 50)
arr7 = np.array([[1, 2], [3, 4]])
arr8 = np.array([[5, 6], [7, 8]])

# 1. Stack two (2, 2) arrays vertically and horizontally
print(np.vstack((arr7, arr8)))
print(np.hstack((arr7, arr8)))

# 2. reshape a (4, 4) array into array of shape (2, 8)
arr9 = np.arange(16).reshape(4, 4)
print(arr9.reshape(2, 8))

# ## 🧠 10. Boolean Masking
# **Question:**
# Given an array:
# 1. Select all elements > 10
# 2. Replace all even numbers with -1


# arr = np.array([[5, 10, 15], [20, 25, 30]])

# ```python
arr = np.array([[5, 10, 15], [20, 25, 30]])
print(arr[arr > 10])
arr[arr % 2 == 0] = -1
print(arr)

# ---

# ## 🧮 11. Aggregation
# **Question:** Calculate:
# 1. Sum of all elements
# 2. Mean of each column
# 3. Max value in each row
# 4. Standard deviation of the array


arr = np.array([[1, 2, 3], [4, 5, 6]])

print(np.sum(arr))
print(np.mean(arr, axis=0))
print(np.max(arr, axis=1))
print(np.std(arr))








# ## 🧲 12. Dot Product and Matrix Multiplication
# **Question:**
# 1. Multiply two matrices using `np.dot`


A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
# 1. Multiply two matrices using `np.dot`
print(np.dot(A, B))









# ---

# ## 🔢 13. Random Numbers
# **Question:**
# 1. Generate a 2x3 array of random numbers between 0 and 1
# 2. Generate 10 random integers between 50 and 100
arr_random = np.random.rand(2, 3)
print(arr_random)
arr_integers = np.random.randint(50, 100, size=10)
print(arr_integers)






# ---

# ## 📐 14. Linear Algebra
# **Question:**
# 1. Find the rank of a matrix
# 2. Compute the inverse of a matrix
# 3. Compute the determinant of a matrix



A = np.array([[1, 2], [3, 4]])
# 1. Find the rank of a matrix
print(np.linalg.matrix_rank(A))
# 2. Compute the inverse of a matrix
print(np.linalg.inv(A))
# 3. Compute the determinant of a matrix
print(np.linalg.det(A))

# ## ⭐ Bonus Challenge
# **Question:**
# Given a 2D array of shape (6, 6), extract all 3x3 submatrices (using slicing) and store them in a list.
print("#"* 50)
arr0 = np.arange(36).reshape(6, 6)
submatrices = []
for i in range(4):
    for j in range(4):
        submatrices.append(arr0[i:i+3, j:j+3])
print(submatrices)

# ---

# Happy coding! 🎯



