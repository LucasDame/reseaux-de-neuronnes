import torch

def sigmoid(x: torch.tensor) -> torch.tensor:
    return 1 / (1 + torch.exp(-x))

class reseaux_en_strates:

    def __init__(self,nom : str, nb_couches: int, nb_lignes: int, taille_entree: int, taille_sortie: int):
        self.nom = nom
        self.nb_couches = nb_couches
        self.nb_lignes = nb_lignes
        self.taille_entree = taille_entree
        self.taille_sortie = taille_sortie

        self.mats_coeffs = [torch.randn(self.nb_lignes, self.nb_lignes, requires_grad=True) 
                            for _ in range(self.nb_couches - 1)]
        self.vec_biais = [torch.zeros(self.nb_lignes, requires_grad=True) for _ in range(self.nb_couches - 1)]
        self.mat_entree = torch.randn(self.taille_entree, self.nb_lignes, requires_grad=True)
        self.mat_sortie = torch.randn(self.nb_lignes, self.taille_sortie, requires_grad=True)
    
    def __repr__(self):
        return (f"{self.nom}(nb_couches={self.nb_couches}, nb_lignes={self.nb_lignes}, "
                f"taille_entree={self.taille_entree}, taille_sortie={self.taille_sortie})")

    def get_mat_entree(self) -> torch.tensor:
        return self.mat_entree
    
    def get_mat_sortie(self) -> torch.tensor:
        return self.mat_sortie
    
    def get_mats_coeffs(self) -> list:
        return self.mats_coeffs
    
    def get_vec_biais(self) -> list:
        return self.vec_biais
    
    def get_nb_couches(self) -> int:
        return self.nb_couches
    
    def get_nb_lignes(self) -> int:
        return self.nb_lignes
    
    def get_taille_entree(self) -> int:
        return self.taille_entree
    
    def get_taille_sortie(self) -> int:
        return self.taille_sortie

    def set_mat_entree(self, mat_entree : torch.tensor):
        if mat_entree.shape[0] != self.taille_entree or mat_entree.shape[1] != self.nb_lignes:
            raise ValueError("La matrice d'entrée doit avoir la forme (taille_entree, nb_lignes)")
        self.mat_entree = mat_entree
    
    def set_mat_sortie(self, mat_sortie : torch.tensor):
        if mat_sortie.shape[0] != self.nb_lignes or mat_sortie.shape[1] != self.taille_sortie:
            raise ValueError("La matrice de sortie doit avoir la forme (nb_lignes, taille_sortie)")
        self.mat_sortie = mat_sortie
    
    def set_mat_coeffs(self, mat_coeffs : list):
        if len(mat_coeffs) != self.nb_couches - 1:
            raise ValueError("La liste des matrices de coefficients doit avoir une longueur égale à nb_couches - 1")
        for i, mat in enumerate(mat_coeffs):
            if mat.shape[0] != self.nb_lignes or mat.shape[1] != self.nb_lignes:
                raise ValueError(f"La matrice de coefficients {i} doit avoir la forme (nb_lignes, nb_lignes)")
            self.mats_coeffs[i] = mat
    
    def set_vec_biais(self, vec_biais : list):
        if len(vec_biais) != self.nb_couches - 1:
            raise ValueError("La liste des biais doit avoir une longueur égale à nb_couches - 1")
        for i, biais in enumerate(vec_biais):
            if biais.shape[0] != self.nb_lignes:
                raise ValueError(f"Le vecteur de biais {i} doit avoir la forme (nb_lignes,)")
            self.vec_biais[i] = biais
    
    def __str__(self) -> str:
        return (f"reseaux_en_strates(nb_couches={self.nb_couches}, nb_lignes={self.nb_lignes}, "
                f"taille_entree={self.taille_entree}, taille_sortie={self.taille_sortie})")
    
    def calc(self, entree: torch.Tensor) -> torch.Tensor:
        val = sigmoid(torch.matmul(entree, self.mat_entree))
        for i in range(self.nb_couches - 1):
            val = sigmoid(torch.matmul(val, self.mats_coeffs[i]) + self.vec_biais[i])
        val = torch.matmul(val, self.mat_sortie)
        return val

    def train(self, entree: torch.Tensor, sortie: torch.Tensor, epochs: int = 100, learning_rate: float = 0.01):
        params = [self.mat_entree, self.mat_sortie] + self.mats_coeffs + self.vec_biais
        optimizer = torch.optim.SGD(params, lr=learning_rate)

        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.calc(entree)
            loss = torch.mean((output - sortie) ** 2)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    def generate(self):
        self.mats_coeffs = [torch.randn(self.nb_lignes, self.nb_lignes, requires_grad=True) 
                            for _ in range(self.nb_couches - 1)]
        self.vec_biais = [torch.zeros(self.nb_lignes, requires_grad=True) for _ in range(self.nb_couches - 1)]
        self.mat_entree = torch.randn(self.taille_entree, self.nb_lignes, requires_grad=True)
        self.mat_sortie = torch.randn(self.nb_lignes, self.taille_sortie, requires_grad=True)
    
    def save(self):
        torch.save({
            'nb_couches': self.nb_couches,
            'nb_lignes': self.nb_lignes,
            'taille_entree': self.taille_entree,
            'taille_sortie': self.taille_sortie,
            'mats_coeffs': [mat.detach() for mat in self.mats_coeffs],
            'vec_biais': [biais.detach() for biais in self.vec_biais],
            'mat_entree': self.mat_entree.detach(),
            'mat_sortie': self.mat_sortie.detach()
        }, self.nom + ".pth")

    def load(self):
        checkpoint = torch.load(self.nom + ".pth")
        self.nb_couches = checkpoint['nb_couches']
        self.nb_lignes = checkpoint['nb_lignes']
        self.taille_entree = checkpoint['taille_entree']
        self.taille_sortie = checkpoint['taille_sortie']
        self.mats_coeffs = [mat.requires_grad_() for mat in checkpoint['mats_coeffs']]
        self.vec_biais = [biais.requires_grad_() for biais in checkpoint['vec_biais']]
        self.mat_entree = checkpoint['mat_entree'].requires_grad_()
        self.mat_sortie = checkpoint['mat_sortie'].requires_grad_()