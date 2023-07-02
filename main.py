#!/usr/bin/env python3

from pathlib import Path

import gudhi as gd
from gudhi.representations import Landscape

import numpy as np
import numpy.typing as npt
import pickle as pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = Path('./data/data_acc_rot.dat')


def main() -> None:
    """Entry point of the program."""
    data = load_data(DATA_PATH)
    persistences: list = []
    for pedestrian_index, pedestrian in tqdm(enumerate(data[:3]), desc='pedestrians', total=len(data) - 1):
        for point_cloud_index, point_cloud in tqdm(enumerate(pedestrian), desc='point_clouds', total=len(pedestrian)):
            persistence = create_persistences(point_cloud)
            persistences.append(persistence)
            save_path = Path('output') / f'pedestrain_{pedestrian_index}_point_cloud_{point_cloud_index}.png'
            create_persistence_plot(persistence, save_path)

    # persistence_path = Path('persistences.pickle')
    # with open(persistence_path, 'wb') as file:
    #     pickle.dump(persistences, file)
    # with open(persistence_path, 'rb') as file:
    #     persistences = pickle.load(file)

    bottleneck_distance_matrix = calc_bottleneck_distance_matrix(persistences)
    calculate_pca_matrix(bottleneck_distance_matrix, 2)

    persistence_landscapes_dimension_1 = create_persistence_landscapes(
        persistences,
        dimension=1,
        n_landscapes=5,
        interval=(0., 1.),
        resolution=500
    )

    input_data = persistence_landscapes_dimension_1
    # raw_input_data = np.vstack([point_cloud.reshape((1, -1)) for pedestrian in data[:3] for point_cloud in pedestrian])

    labels = [0] * 100 + [1] * 100 + [2] * 100
    X_train, X_test, y_train, y_test = train_test_split(
        input_data,
        labels,
        test_size=0.2
    )

    classifier = RandomForestClassifier(max_depth=5, random_state=10)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    correct_ratio = (predictions == np.array(y_test)).sum() / len(y_test)
    print(f'The ratio of correctly predicted labels is {correct_ratio}.')


def load_data(data_path: Path):
    """Load the pedestrian data from the specified path."""
    with open(data_path, 'rb') as data_file:
        data = pickle.load(data_file, encoding='latin1')
    return data


def create_persistences(point_cloud: npt.NDArray[np.float_]):
    """Calculate the persistence of a point cloud."""
    rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=2.)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    simplex_tree.compute_persistence()
    persistence = simplex_tree.persistence()
    return persistence


def calculate_pca_matrix(matrix, dimension: int):
    """Apply principal component analysis on a given matrix."""
    pca = PCA(dimension)
    transformed_values_2 = pca.fit_transform(matrix)
    figure = plt.figure()
    axes = figure.add_subplot()
    colors = np.r_[
        1 * np.ones((transformed_values_2.shape[0] // 3)),
        2 * np.ones((transformed_values_2.shape[0] // 3)),
        3 * np.ones((transformed_values_2.shape[0] // 3))
    ]
    axes.scatter(transformed_values_2[:, 0], transformed_values_2[:, 1], c=colors)
    figure.savefig(Path('bottleneck_distance_matrix_pca.png'))


def create_persistence_landscapes(
    persistences,
    dimension,
    n_landscapes,
    interval,
    resolution,
):
    """Write a function to compute persistence landscapes.
    - persistence diagram dgm,
    - dimension
    - number of landscapes to compute
    - endpoints xmin, xmax of interval
    - resolution of the landscapes as a number of points
    """
    landscapes = []
    for persistence in persistences:
        points = np.array([[x, y] for (dim, (x, y)) in persistence if dim == dimension])
        landscape = Landscape(
            num_landscapes=n_landscapes,
            resolution=resolution,
            sample_range=[interval[0], interval[1]]
        ).fit_transform([points])
        landscape[np.isnan(landscape)] = 0
        landscapes.append(landscape[0])
        # landscape = landscape.reshape((-1, resolution)).T
        # plt.plot(np.linspace(interval[0], interval[1], resolution), landscape[:, 0])

    return landscapes


def calc_bottleneck_distance_matrix(persistences):

    result = np.empty((len(persistences), len(persistences)))
    for row, persistence_1 in tqdm(enumerate(persistences), total=len(persistences)):
        persistence_1_dimension_1 = np.array([[x, y] for (dimension, (x, y)) in persistence_1 if dimension == 1])
        for column, persistence_2 in enumerate(persistences):
            persistence_2_dimension_1 = np.array([[x, y] for (dimension, (x, y)) in persistence_2 if dimension == 1])
            result[row, column] = gd.bottleneck_distance(persistence_1_dimension_1, persistence_2_dimension_1)
    return result


def create_persistence_plot(persistence, path: Path) -> None:
    figure, (axes_left, axes_right) = plt.subplots(1, 2)
    gd.plot_persistence_barcode(persistence, axes=axes_left)
    gd.plot_persistence_diagram(persistence, axes=axes_right)
    figure.savefig(path)
    plt.close()


if __name__ == '__main__':
    main()
