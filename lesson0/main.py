# The optimal values of m and b can be actually calculated with way less effort than doing a linear regression.
# this is just to demonstrate gradient descent

from numpy import *


# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))


def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return new_b, new_m


def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations):
    b = initial_b
    m = initial_m
    with open('training_history.csv', 'w') as training_history:
        for i in range(num_iterations):
            b, m = step_gradient(b, m, points, learning_rate)
            current_error = compute_error_for_line_given_points(b, m, points)
            training_history.write(f"{i}, {b}, {m}, {current_error}\n")
    return b, m


def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.00001
    num_iterations = 10000
    initial_b = 0  # initial y-intercept guess
    initial_m = 0  # initial slope guess

    print(f"Starting gradient descent at b = {initial_b}, m = {initial_m}, error = {compute_error_for_line_given_points(initial_b, initial_m, points)}")
    print("Running...")
    b, m = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print(f"After {num_iterations} iterations b = {b}, m = {m}, error = {compute_error_for_line_given_points(b, m, points)}")

if __name__ == '__main__':
    run()
