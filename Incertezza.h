#ifndef INCERTEZZA_H
#define INCERTEZZA_H

// Stati di occupazione
#define STATO_AWAY   0
#define STATO_HOME   1
#define STATO_SLEEP  2
#define N_STATI      3

/* --------------------------------------------------
 * Funzione di utilità U(s)
 *
 * stato     : AWAY / HOME / SLEEP
 * temp_int  : temperatura interna (°C)
 * temp_ext  : temperatura esterna (°C)
 *
 * Ritorna il valore di utilità associato allo stato
 * -------------------------------------------------- */
double calcola_utilita(
    int stato,
    double temp_int,
    double temp_ext
);

/* --------------------------------------------------
 * Utilità Attesa
 *
 * p[]       : distribuzione P(stato | evidenze)
 * temp_int  : temperatura interna
 * temp_ext  : temperatura esterna
 *
 * EU = Σ p[s] * U(s)
 * -------------------------------------------------- */
double utilita_attesa(
    const double p[],
    double temp_int,
    double temp_ext
);

#endif
