#ifndef PL_SCHEDULER_H
#define PL_SCHEDULER_H

/* ============================================================
 *              MACROAREA DECISIONE (ICON3)
 * ============================================================
 *
 * Modulo che implementa la fase finale del sistema:
 * la DECISIONE OTTIMALE.
 *
 * Utilizza la Programmazione Lineare (PL) per determinare
 * come distribuire la potenza di riscaldamento sugli slot
 * temporali / appartamenti, rispettando vincoli di costo
 * e rischio.
 */

/* Numero massimo di slot gestibili dal pianificatore */
#define MAX_SLOTS 10

/* ============================================================
 * STRUTTURA RISULTATO DELLA PL
 *
 * power[i] ∈ [0,1] rappresenta il livello ottimale di
 * riscaldamento assegnato allo slot i-esimo.
 *
 * n indica il numero effettivo di slot utilizzati.
 * ============================================================ */
typedef struct {
    double power[MAX_SLOTS]; // livello di riscaldamento (0 = spento, 1 = massimo)
    int n;                   // numero di slot considerati
} PL_Risultato;

/* ============================================================
 * FUNZIONE DI OTTIMIZZAZIONE
 *
 * Calcola il piano energetico ottimale massimizzando
 * il beneficio complessivo (comfort) sotto vincoli
 * di costo e rischio.
 *
 * INPUT:
 *  - occ_prob[i]     : probabilità che lo slot sia occupato
 *  - price[i]        : costo dell'energia nello slot i
 *  - comfort_gain[i] : beneficio (utilità attesa) per unità di potenza
 *  - risk_coeff[i]   : coefficiente di rischio (incertezza)
 *  - n               : numero di slot
 *  - budget          : costo massimo consentito
 *  - risk_max        : rischio massimo accettabile
 *
 * OUTPUT:
 *  - struttura PL_Risultato con le potenze ottimali
 * ============================================================ */
PL_Risultato calcolarePianoOttimale(
    const double occ_prob[],     // P(slot occupato)
    const double price[],        // costo energia per slot
    const double comfort_gain[], // beneficio comfort (utilità attesa)
    const double risk_coeff[],   // rischio associato allo slot
    int n,
    double budget,               // budget massimo di costo
    double risk_max              // rischio massimo consentito
);

#endif
