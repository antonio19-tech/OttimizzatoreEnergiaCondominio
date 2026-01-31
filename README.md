# Ottimizzatore Energetico Condominio

Sistema di ottimizzazione energetica basato su:
- apprendimento automatico probabilistico
- ragionamento sotto incertezza (utilità attesa)
- programmazione lineare

---

## 📌 Descrizione del sistema

Il sistema è composto da tre macro-aree:

1. **Apprendimento Automatico**
   - Rete neurale feed-forward supervisionata
   - Output probabilistico sugli stati di occupazione:
     - Away
     - Home
     - Sleep

2. **Incertezza e Utilità Attesa**
   - Calcolo dell’utilità attesa a partire dalle probabilità apprese
   - Funzione di utilità definita dal dominio

3. **Programmazione Lineare**
   - Ottimizzazione del piano energetico
   - Vincoli di budget e rischio
   - Risoluzione tramite GLPK

---

## 🗂 Struttura del progetto
├── src/
│ ├── NeuralNetwork.c /.h
│ ├── Incertezza.c /.h
│ ├── PL_Scheduler.c /.h
│ └── main.c
├── dataset.csv
├── Makefile
├── Documentazione.pdf
└── README.md

## ⚙️ Compilazione ed esecuzione

Requisiti:
- GCC
- GLPK

Compilazione:
make

Esecuzione: 
./main
