import csv
import matplotlib.pyplot as plt


class SimpleLinearRegression:

    def __init__(self, csv_file):
        self.csv_file = csv_file

        self.slope_coef = None
        self.y_intercept = None

        self.num_lines = 0

        self.X_values = []
        self.y_values = []

        with open(self.csv_file, mode='r') as file:
            self.csv_file = csv.reader(file)
            for index, lines in enumerate(self.csv_file):
                if index == 0:
                    self.y_label = lines[0]
                    self.X_label = lines[1]
                    continue

                self.X_values.append(float(lines[0]))
                self.y_values.append(float(lines[1]))
                self.num_lines += 1

        self.X_mean = sum(self.X_values) / self.num_lines
        self.y_mean = sum(self.y_values) / self.num_lines

    def fit(self):
        """
        Solve for slope coefficient `slope_coef` and y-intercept
        `y-intercept`

        m = sum((xi - x̄)(yi - ȳ))/sum((xi - x̄) ** 2)
        """
        numerator_summation = 0
        denominator_summation = 0
        for x_value, y_value in zip(self.X_values, self.y_values):
            #                    sum((xi - x̄) * (yi - ȳ))
            numerator_summation +=   (x_value - self.X_mean) * \
                                     (y_value - self.y_mean)
            #                    sum((xi - x̄) ** 2)
            denominator_summation += (x_value - self.X_mean) ** 2

        #             m = sum((xi - x̄)(yi - ȳ))/sum((xi - x̄) ** 2)
        self.slope_coef = numerator_summation / denominator_summation
        #              b = ȳ - mx̄
        self.y_intercept = self.y_mean - self.slope_coef * self.X_mean

        print(f"y = {self.y_intercept} + {self.slope_coef}(X)")

    def predict(self, prediction_value: float):
        """
        Predict `prediction_value` using fitted model
        """
        if prediction_value:
            if (self.slope_coef is not None and
                self.y_intercept is not None):
                prediction = (self.y_intercept
                            + self.slope_coef
                            * prediction_value)
                return prediction
            return None
        return None

    def plot_points_and_line(self):
        y_prediction_values = []
        for value in self.X_values:
            y_prediction_values.append(self.predict(value))

        plt.scatter(self.X_values, self.y_values, color="red")
        plt.plot(self.X_values, y_prediction_values, color="blue")
        plt.xlabel(self.X_label)
        plt.ylabel(self.y_label)
        plt.title("Simple Linear Regression")
        plt.show()