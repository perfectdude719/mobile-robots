import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data

# Visualization functions
def plot_state(mu, sigma, landmarks, map_limits):
    lx, ly = zip(*[landmarks[i] for i in sorted(landmarks.keys())])
    covariance = sigma[:2, :2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)
    max_ind = np.argmax(eigenvals)
    angle = np.arctan2(eigenvecs[1, max_ind], eigenvecs[0, max_ind])
    width, height = 2 * np.sqrt(2.2789 * eigenvals)

    ell = Ellipse(xy=(mu[0], mu[1]), width=width, height=height, angle=np.degrees(angle), alpha=0.25)
    plt.clf()
    plt.gca().add_artist(ell)
    plt.plot(lx, ly, 'bo', markersize=10)
    plt.quiver(mu[0], mu[1], np.cos(mu[2]), np.sin(mu[2]))
    plt.axis(map_limits)
    plt.pause(0.01)

# EKF Functions
def prediction_step(odometry, mu, sigma, R):
    delta_rot1, delta_trans, delta_rot2 = odometry.values()
    x, y, theta = mu

    # Motion model
    mu[0] += delta_trans * np.cos(theta + delta_rot1)
    mu[1] += delta_trans * np.sin(theta + delta_rot1)
    mu[2] += delta_rot1 + delta_rot2

    # Normalize theta
    mu[2] = (mu[2] + np.pi) % (2 * np.pi) - np.pi

    # Jacobian G_t
    G_t = np.eye(3)
    G_t[0, 2] = -delta_trans * np.sin(theta + delta_rot1)
    G_t[1, 2] = delta_trans * np.cos(theta + delta_rot1)

    # Update covariance
    sigma = G_t @ sigma @ G_t.T + R
    return mu, sigma


def correction_step(sensor_data, mu, sigma, landmarks, Q):
    # Ensure mu is a float array
    mu = mu.astype(float)
    
    x, y, theta = mu
    ids, ranges = sensor_data['id'], sensor_data['range']

    for i, landmark_id in enumerate(ids):
        lx, ly = landmarks[landmark_id]
        expected_range = np.sqrt((lx - x)**2 + (ly - y)**2)

        # Measurement Jacobian H_t
        H_t = np.zeros((1, 3))
        H_t[0, 0] = (x - lx) / expected_range
        H_t[0, 1] = (y - ly) / expected_range

        # Kalman gain
        S = H_t @ sigma @ H_t.T + Q
        K_t = sigma @ H_t.T @ np.linalg.inv(S)

        # Update state
        z = ranges[i]
        mu += (K_t @ np.array([[z - expected_range]])).flatten()
        sigma = (np.eye(3) - K_t @ H_t) @ sigma

    return mu, sigma



# Main function
def main():
    landmarks = read_world("world.dat")
    sensor_data = read_sensor_data("sensor_data.dat")

    # Initial state and covariance
    mu = np.array([0, 0, 0])
    sigma = np.eye(3)
    R = np.diag([0.1, 0.1, 0.05])  # Motion noise
    Q = np.array([[0.1]])  # Measurement noise
    map_limits = [-1, 12, -1, 10]

    for t in range(len(sensor_data) // 2):
        plot_state(mu, sigma, landmarks, map_limits)
        mu, sigma = prediction_step(sensor_data[t, 'odometry'], mu, sigma, R)
        mu, sigma = correction_step(sensor_data[t, 'sensor'], mu, sigma, landmarks, Q)

    plt.show()

if __name__ == "__main__":
    main()
