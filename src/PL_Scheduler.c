#include <glpk.h>
#include "PL_Scheduler.h"

/* ============================================================
 *              MACROAREA DECISIONE (ICON3)
 * ============================================================
 *
 * Questo modulo traduce le informazioni probabilistiche
 * e l’utilità attesa (calcolata nei moduli precedenti)
 * in una decisione operativa ottimale.
 *
 * Il problema è formulato come un problema di
 * Programmazione Lineare (PL) risolto con GLPK.
 *
 * Variabili decisionali:
 *   x_i ∈ [0,1] → livello di potenza del riscaldamento
 *                 per lo slot/appartamento i-esimo.
 *
 * Funzione obiettivo:
 *   massimizzare la somma delle utilità attese,
 *   tenendo conto del costo dell’energia.
 *
 * Vincoli:
 * 1) Vincolo di budget energetico
 * 2) Vincolo di rischio complessivo
 */

/*
 * ============================================================
 * FUNZIONE calcolarePianoOttimale
 *
 * Costruisce e risolve il problema di PL:
 *
 *   max  Σ x_i * (occ_prob[i] * comfort_gain[i] - price[i])
 *
 * soggetto a:
 *   Σ x_i * price[i]      ≤ budget
 *   Σ x_i * risk_coeff[i] ≤ risk_max
 *   0 ≤ x_i ≤ 1
 *
 * ============================================================
 */
PL_Risultato calcolarePianoOttimale(
    const double occ_prob[],     // Probabilità di occupazione per slot
    const double price[],        // Prezzo dell’energia per slot
    const double comfort_gain[], // Utilità attesa per unità di potenza
    const double risk_coeff[],   // Coefficiente di rischio
    int n,                       // Numero di slot
    double budget,               // Budget massimo consentito
    double risk_max              // Rischio massimo accettabile
){
    /* Struttura che conterrà il risultato finale */
    PL_Risultato res;
    res.n = n;

    /* Inizializzazione: potenza nulla per tutti gli slot */
    for (int i = 0; i < MAX_SLOTS; i++)
        res.power[i] = 0.0;

    /* Caso limite: nessuno slot */
    if (n <= 0) return res;

    /* ========================================================
     * CREAZIONE DEL PROBLEMA DI PROGRAMMAZIONE LINEARE
     * ======================================================== */
    glp_prob *lp = glp_create_prob();
    glp_set_prob_name(lp, "heating_schedule");
    glp_set_obj_dir(lp, GLP_MAX); // Massimizzazione

    /* ========================================================
     * DEFINIZIONE DELLE VARIABILI DECISIONALI x_i
     * ======================================================== */
    glp_add_cols(lp, n);
    for (int i = 1; i <= n; i++) {

        /* Vincolo sulle variabili: 0 ≤ x_i ≤ 1 */
        glp_set_col_bnds(lp, i, GLP_DB, 0.0, 1.0);

        /* Variabile continua */
        glp_set_col_kind(lp, i, GLP_CV);

        /* Coefficiente della funzione obiettivo:
         * utilità attesa - costo */
        double coef = occ_prob[i-1] * comfort_gain[i-1] - price[i-1];
        glp_set_obj_coef(lp, i, coef);
    }

    /* ========================================================
     * DEFINIZIONE DEI VINCOLI
     * ======================================================== */
    glp_add_rows(lp, 2);

    /* Vincolo 1: budget energetico */
    glp_set_row_bnds(lp, 1, GLP_UP, 0.0, budget);

    /* Vincolo 2: rischio massimo */
    glp_set_row_bnds(lp, 2, GLP_UP, 0.0, risk_max);

    int ind[MAX_SLOTS + 1];
    double val[MAX_SLOTS + 1];

    /* --- Vincolo di budget: Σ x_i * price[i] ≤ budget --- */
    for (int i = 0; i < n; i++) {
        ind[i+1] = i + 1;
        val[i+1] = price[i];
    }
    glp_set_mat_row(lp, 1, n, ind, val);

    /* --- Vincolo di rischio: Σ x_i * risk_coeff[i] ≤ risk_max --- */
    for (int i = 0; i < n; i++) {
        ind[i+1] = i + 1;
        val[i+1] = risk_coeff[i];
    }
    glp_set_mat_row(lp, 2, n, ind, val);

    /* ========================================================
     * RISOLUZIONE DEL PROBLEMA
     * ======================================================== */
    glp_smcp parm;
    glp_init_smcp(&parm);

    /* Disattivazione output del simplex */
    parm.msg_lev = GLP_MSG_OFF;

    glp_simplex(lp, &parm);

    /* ========================================================
     * ESTRAZIONE DELLA SOLUZIONE OTTIMA
     * ======================================================== */
    for (int i = 0; i < n; i++) {
        double x = glp_get_col_prim(lp, i + 1);
        res.power[i] = (x > 0.0) ? x : 0.0;
    }

    /* Pulizia memoria */
    glp_delete_prob(lp);

    return res;
}
