import csv


class MultipleLinearRegression:

    def __init__(self, csv_file):
        self.coefficients = []
        self.csv_file = csv_file

        self.X_matrix = []
        self.y_vector = []

        with open(self.csv_file, mode='r') as file:
            self.csv_file = csv.reader(file)
            for index, lines in enumerate(self.csv_file):
                if index == 0:
                    self.X_label = lines[:-1]
                    self.y_label = lines[-1]
                    continue

                self.X_matrix.append(
                    [1] + [float(x) for x in lines[:-1]]
                )
                self.y_vector.append(
                    float(lines[-1])
                )

        self.X_matrix_transmuted = [
            list(row) for row in zip(*self.X_matrix)
        ]

    @staticmethod
    def _xtx_multiply(matrix_a: list, matrix_b: list):
        num_row_a = len(matrix_a)
        num_col_b = len(matrix_b[0])

        product_size = (num_row_a, num_col_b)

        product_matrix = [
            [0 for x in range(num_col_b)] for x in range(num_row_a)
        ]

        num_ops = len(matrix_a[0])
        for num_row in range(product_size[0]):
            for num_col in range(product_size[1]):
                cell_sum = 0

                multiply_a = matrix_a[num_row]
                multiply_b = [x[num_col] for x in matrix_b]

                for num_cell in range(num_ops):
                    cell_sum += (
                            multiply_a[num_cell] * multiply_b[num_cell]
                    )

                product_matrix[num_row][num_col] += cell_sum

        return product_matrix

    @staticmethod
    def _xty_multiply(matrix_a: list, matrix_b: list):
        return [
            sum([x[0] * x[1] for x in zip(row, matrix_b)])
            for row in matrix_a
        ]

    @staticmethod
    def _gaussian_eliminate(matrix_a: list, matrix_b: list):
        for cell_index, cell in enumerate(matrix_b):
            matrix_a[cell_index].append(cell)

        matrix = [
            [] for x in range(len(matrix_a))
        ]

        cur_pivot_row = 0
        cur_pivot_col = 0

        for num_col in range(len(matrix)):
            if cur_pivot_row == 0 and cur_pivot_col == 0:
                pivot = matrix_a[cur_pivot_row][cur_pivot_col]
                if pivot == 0:
                    cur_pivot_row += 1
                matrix[cur_pivot_row] = [
                    x / pivot for x in matrix_a[cur_pivot_row]
                ]
                for num_row, row in enumerate(matrix_a):
                    if num_row <= cur_pivot_row:
                        continue
                    subtrahend = [
                        x * matrix_a[num_row][num_col] for x in matrix[cur_pivot_row]
                    ]
                    matrix[num_row] = [
                        x - y for x, y in zip(matrix_a[num_row], subtrahend)
                    ]
            else:
                pivot = matrix[cur_pivot_row][cur_pivot_col]
                if pivot == 0:
                    cur_pivot_row += 1
                matrix[cur_pivot_row] = [
                    x / pivot for x in matrix[cur_pivot_row]
                ]
                for num_row, row in enumerate(matrix):
                    if num_row <= cur_pivot_row:
                        continue
                    subtrahend = [
                        x * matrix[num_row][num_col] for x in matrix[cur_pivot_row]
                    ]
                    matrix[num_row] = [
                        x - y for x, y in zip(matrix[num_row], subtrahend)
                    ]
            cur_pivot_row += 1
            cur_pivot_col += 1

        coefficients = [0] * len(matrix)

        for i in range(len(matrix) - 1, -1, -1):
            rhs = matrix[i][-1]
            for j in range(i + 1, len(matrix)):
                rhs -= matrix[i][j] * coefficients[j]
            coefficients[i] = rhs

        return coefficients

    def fit(self):
        xtx = self._xtx_multiply(
            self.X_matrix_transmuted, self.X_matrix
        )

        xty = self._xty_multiply(
            self.X_matrix_transmuted, self.y_vector
        )

        self.coefficients = self._gaussian_eliminate(xtx, xty)
        print(self.coefficients)

    def predict(self, prediction_values: list):
        if self.coefficients:
            return sum([
                x * y for x, y in zip(
                    self.coefficients, prediction_values
                )
            ])
        else:
            print("Model not fitted")
        return None


if __name__ == "__main__":
    data_sheet = MultipleLinearRegression("exam_score.csv")
    data_sheet.fit()
    print(data_sheet.predict([1, 11, 7, 20]))
