#include "Incertezza.h"

/* ============================================================
 *                MACROAREA INCERTEZZA (ICON9)
 * ============================================================
 *
 * Questo modulo implementa il ragionamento decisionale sotto
 * incertezza tramite il concetto di UTILITÀ ATTESA.
 *
 * La rete neurale fornisce una distribuzione di probabilità
 * sugli stati di occupazione:
 *
 *   P(Away), P(Home), P(Sleep)
 *
 * Questo modulo assegna un valore di utilità a ciascuno stato
 * in funzione delle condizioni ambientali (temperatura interna
 * ed esterna) e combina tali valori tramite l'utilità attesa.
 */

/* ============================================================
 * FUNZIONE DI UTILITÀ U(s)
 *
 * Associa un valore di utilità a ciascuno stato di occupazione
 * in base alle condizioni ambientali.
 *
 * stato    : stato di occupazione (Away, Home, Sleep)
 * temp_int : temperatura interna
 * temp_ext : temperatura esterna
 *
 * Ritorna:
 *   Valore numerico che rappresenta il "beneficio" o "costo"
 *   di riscaldare in quello stato.
 * ============================================================ */
double calcola_utilita(
    int stato,
    double temp_int,
    double temp_ext
) {
    double u = 0.0;

    switch (stato) {

        case STATO_AWAY:
            /*
             * Stato AWAY:
             * - Nessuna presenza in casa
             * - Obiettivo principale: risparmio energetico
             * - Penalità se si spreca energia mantenendo
             *   una temperatura interna elevata
             */
            if (temp_int > 16.0)
                u = -0.5;
            else
                u = 0.0;
            break;

        case STATO_HOME:
            /*
             * Stato HOME:
             * - Presenza attiva in casa
             * - Comfort prioritario
             * - Alta utilità se la temperatura è sotto
             *   la soglia di comfort
             */
            if (temp_int < 19.5)
                u = 1.2;
            else
                u = 0.4;

            /*
             * Bonus contestuale:
             * Se fuori fa freddo, il valore del comfort
             * interno aumenta ulteriormente
             */
            if (temp_ext < 8.0 && temp_int < 20.0)
                u += 0.5;
            break;

        case STATO_SLEEP:
            /*
             * Stato SLEEP:
             * - Comfort moderato
             * - Penalità forte se la casa è troppo calda
             * - Leggera utilità se la temperatura è bassa
             *   in condizioni di freddo esterno
             */
            if (temp_int >= 19.5)
                u = -2.5;
            else
                u = 0.2;

            /*
             * Bonus notturno:
             * Se fuori e dentro fa molto freddo,
             * mantenere un minimo di tepore è utile
             */
            if (temp_ext < 5.0 && temp_int < 17.0)
                u += 0.3;
            break;

        default:
            // Stato non valido → utilità nulla
            u = 0.0;
    }

    return u;
}

/* ============================================================
 * UTILITÀ ATTESA (Expected Utility)
 *
 * EU = Σ P(s) * U(s)
 *
 * p[]      : distribuzione di probabilità sugli stati
 * temp_int : temperatura interna
 * temp_ext : temperatura esterna
 *
 * Ritorna:
 *   Valore scalare che rappresenta la convenienza attesa
 *   di attivare il riscaldamento in questo scenario.
 * ============================================================ */
double utilita_attesa(
    const double p[],
    double temp_int,
    double temp_ext
) {
    double eu = 0.0;

    /*
     * Combinazione probabilistica:
     * ogni utilità viene pesata con la probabilità
     * stimata dalla rete neurale
     */
    for (int s = 0; s < N_STATI; s++) {
        eu += p[s] * calcola_utilita(s, temp_int, temp_ext);
    }

    return eu;
}
