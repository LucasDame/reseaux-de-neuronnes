import torch

class reseaux_en_strates:

    def __init__(self, nb_couches : int, nb_lignes : int, taille_entree : int, taille_sortie : int):
        self.nb_couches = nb_couches
        self.nb_lignes = nb_lignes
        self.taille_entree = taille_entree
        self.taille_sortie = taille_sortie
        self.mats_coeffs = torch.tesor([[[]x]for i in nb_couches-1])
        self.mat_entrée = 
        self.mat_sortie = 
        self.biais = []