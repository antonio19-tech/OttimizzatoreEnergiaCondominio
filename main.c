#include <stdio.h>
#include <stdlib.h>
#include "NeuralNetwork.h"
#include "Incertezza.h"
#include "PL_Scheduler.h"

/* ============================================================
 * PARAMETRI GLOBALI DEL SISTEMA
 * ============================================================ */
#define N_SLOTS         4       // Numero di appartamenti / slot decisionali
#define N_FEATURES      7       // Numero di feature di input della rete
#define N_STATI         3       // Stati: Away, Home, Sleep
#define EPOCHE          500     // Epoche di addestramento rete neurale
#define BUDGET          1.2     // Vincolo massimo di energia consumabile
#define RISCHIO         0.1     // Vincolo massimo di rischio globale

/* ============================================================
 * MACROAREA 1 — APPRENDIMENTO (ICON7–ICON8)
 * Rete Neurale con output probabilistico
 * ============================================================ */

/*
 * Carica il dataset da file CSV ed esegue l'addestramento
 * della rete neurale.
 *
 * - Normalizza le feature (feature engineering)
 * - La rete impara P(Stato | Evidenze)
 */
void train_system(NeuralNetwork *net, const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) return;

    double row[N_FEATURES];
    int target_class;

    while (fscanf(
        f,
        "%lf, %lf, %lf, %lf, %lf, %lf, %lf, %d",
        &row[0], &row[1], &row[2],
        &row[3], &row[4], &row[5],
        &row[6], &target_class
    ) == 8) {

        // Normalizzazione delle feature (ICON8)
        double input[N_FEATURES] = {
            row[0] / 24.0,   // ora
            row[1] / 10.0,   // temperatura esterna
            row[2],          // luci
            row[3],          // movimento
            row[4] / 10.0,   // consumo
            row[5],          // prezzo energia
            row[6] / 30.0    // temperatura interna
        };

        // Target one-hot (Away, Home, Sleep)
        double target[3] = {0, 0, 0};
        target[target_class] = 1.0;

        nn_train(net, input, target);
    }

    fclose(f);
}

/* ============================================================
 * MAIN
 * ============================================================ */
int main(void) {

    srand(42);

    /* ========================================================
     * MACROAREA 1 — APPRENDIMENTO
     * ======================================================== */

    // Creazione della rete neurale
    NeuralNetwork *ann = nn_create(
        N_FEATURES,   // input
        16,           // neuroni hidden
        3,            // output probabilistici
        0.01,         // learning rate
        0.001         // regolarizzazione L2
    );

    // Addestramento su dataset
    for (int e = 0; e < EPOCHE; e++)
        train_system(ann, "dataset.csv");

    /* ========================================================
     * MACROAREA 2 — INCERTEZZA / VALUTAZIONE STOCASTICA (ICON9)
     * ======================================================== */

    // Dati di test (stato corrente degli appartamenti)
    double slots_test[N_SLOTS][N_FEATURES] = {
        // ora, temp_ext, luci, mov, consumo, prezzo, temp_int
        {19, 6, 0.8, 0.3, 3.0, 0.50, 16.0},
        {19, 6, 0.7, 0.5, 3.0, 0.47, 17.0},
        {19, 5, 0.0, 0.0, 0.5, 0.42, 16.0},
        {19, 5, 0.5, 0.6, 5.5, 0.45, 18.0}
    };

    double comfort_gain[N_SLOTS];
    double prices[N_SLOTS];
    double risk_coeff[N_SLOTS];
    double occ_prob[N_SLOTS];

    printf("\n\n--- ANALISI AGENTE INTELLIGENTE ---\n\n");

    for (int i = 0; i < N_SLOTS; i++) {

        // Normalizzazione input per inferenza
        double input_norm[N_FEATURES] = {
            slots_test[i][0] / 24.0,
            slots_test[i][1] / 10.0,
            slots_test[i][2],
            slots_test[i][3],
            slots_test[i][4] / 5.0,
            slots_test[i][5],
            slots_test[i][6] / 30.0
        };

        // Inferenza neurale: P(Stato | Evidenze)
        nn_forward(ann, input_norm);
        double *p = ann->output;

        double t_int = slots_test[i][6];
        double t_ext = slots_test[i][1];

        // Utilità Attesa (ICON9)
        double eu = utilita_attesa(p, t_int, t_ext);

        comfort_gain[i] = eu;
        prices[i] = slots_test[i][5];
        risk_coeff[i] = p[0];           // Rischio = P(Away)
        occ_prob[i] = p[1] + p[2];      // Presenza

        printf(
            "Appartamento %d:\n"
            "ORA[%.0f:00] T_EXT[%.0f°] T_INT[%.0f°] LUCI[%.1f] MOVIMENTO[%.1f]->\n"
            "P(Away): %.2f | P(Home): %.2f | P(Sleep): %.2f | EU Totale: %.3f\n\n",
            i + 1,
            slots_test[i][0],
            slots_test[i][1],
            slots_test[i][6],
            slots_test[i][2],
            slots_test[i][3],
            p[0], p[1], p[2], eu
        );
    }

    /* ========================================================
     * MACROAREA 3 — DECISIONE OTTIMALE (PL - ICON3)
     * ======================================================== */

    PL_Risultato piano = calcolarePianoOttimale(
        occ_prob,
        prices,
        comfort_gain,
        risk_coeff,
        N_SLOTS,
        BUDGET,
        RISCHIO
    );

    printf("\n--- PIANO ENERGETICO OTTIMALE ---\n");
    for (int i = 0; i < N_SLOTS; i++)
        printf(
            "Appartamento %d -> Potenza %.1f%%\n",
            i + 1,
            piano.power[i] * 100
        );

    nn_free(ann);
    return 0;
}
