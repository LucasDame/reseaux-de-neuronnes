import struct
import numpy as np
import torch

def lire_images_mnist(fichier: str) -> np.ndarray:
    with open(fichier, 'rb') as f:
        magic, nb_images, nb_lignes, nb_colonnes = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError("Ce fichier n'est pas un fichier d'images MNIST valide.")
        taille = nb_images * nb_lignes * nb_colonnes
        images = np.frombuffer(f.read(taille), dtype=np.uint8)
        images = images.reshape((nb_images, nb_lignes, nb_colonnes))
        images = images / 255.0  # Normalisation entre 0 et 1
        return images / 255.0  # Normalisation entre 0 et 1

def lire_labels_mnist(fichier: str) -> np.ndarray:
    with open(fichier, 'rb') as f:
        magic, nb_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError("Ce fichier n'est pas un fichier de labels MNIST valide.")
        labels = np.frombuffer(f.read(nb_labels), dtype=np.uint8)
        return labels

def labels_to_sorties(labels: np.ndarray) -> torch.tensor:
    nb_labels = len(labels)
    sorties = torch.zeros((nb_labels, 10), dtype=torch.float32)
    for i, label in enumerate(labels):
        sorties[i, label] = 1.0
    return sorties

def images_to_entrees(images: np.ndarray) -> torch.tensor:
    nb_images, nb_lignes, nb_colonnes = images.shape
    images = images.reshape((nb_images, nb_lignes * nb_colonnes))
    return torch.tensor(images, dtype=torch.float32).t()  # Transpose pour correspondre à la taille d'entrée