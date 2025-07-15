import Reseaux_de_neuronnes as rn
import Dataset_chiffres as dc
import os
import torch



start_over = False # Si True, recommence l'entraînement depuis le début

if __name__ == "__main__":
    reco_chiffres = rn.reseaux_en_strates(nom="Reconnaissance de Chiffres",
                                        nb_couches=3,
                                        nb_lignes=10,
                                        taille_entree=784,
                                        taille_sortie=10)
    if os.path.isfile("Reconnaissance_de_chiffres.pth") and not start_over:
        reco_chiffres.load()
    else:
        reco_chiffres.generate()
    images = dc.images_to_entrees(dc.lire_images_mnist("train-images-idx3-ubyte"))
    labels = dc.labels_to_sorties(dc.lire_labels_mnist("train-labels-idx1-ubyte"))
    for _ in range(len(labels)):
        reco_chiffres.train(images, labels, epochs=100, learning_rate=0.01)
    reco_chiffres.save()
    print(reco_chiffres)