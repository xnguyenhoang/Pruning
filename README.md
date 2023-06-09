# Analýza efektivity pruningových algoritmov
![](https://img.shields.io/badge/PyTorch-v.%202.0.1-orange)

Programy sa používa na testovanie efektivity pruningových alrgoritmov
Potrebné knižnice na spusteni:<br />
- Pytorch v. 2.0.1  pip3 install torch torchvision torchaudio

V repozitári sa nachádza 2 python súbory, MNIST dátový balík a bakalársky projekt v pdf forme.

Stiahneme si repozitár s projektom.

fnn.py obsahuje doprednú neurónovú sieť.
cmm.py ubsahuje konvolučnú neurónovú sieť
Otvoríme si projekt vo Vami zvoledom vývojarskom prostredí (IDE).
Na začiatku projektov je zakomentovaný "torch.manual_seed(10)", ktorý slúži na kontrolovanú generáciu "náhodných" čisel pre zabezpečenie reprodukovateľnosti výsledkov.

Po spustení projektu sa stiahne dátový balík MNIST do súboru data, ak ešte náhodou nie je stiahnutý, inicializuje sa neurónová sieť a začne sa porovnávanie pruningových algoritmov.

Odkaz na GutHub repozitár: https://github.com/xnguyenhoang/Pruning
